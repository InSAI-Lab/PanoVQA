# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import plm.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from plm.data.data_qwen import make_supervised_data_module
from plm.data.data_qwen_packed import make_supervised_data_module_packed
from plm.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    AdaptorArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig
from torch import nn
# from adaptorformer import modify_qwen2_5_vl_visionblock
from attentionFormer import modify_qwen2_5_vl_vision_attention
from logger import setup_logger


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_trainable_status(module: nn.Module, status: bool):
    """Set the requires_grad status for all parameters in a module."""
    for name, param in module.named_parameters():
        param.requires_grad = status

def set_model(model_args, model):
    """Configure which parts of the model are trainable based on model_args."""
    # Freeze or unfreeze main components
    print("train model visual: ", model_args.tune_mm_vision)
    set_trainable_status(model.visual, model_args.tune_mm_vision)

    print("train model visual merger: ", model_args.tune_mm_mlp)
    set_trainable_status(model.visual.merger, model_args.tune_mm_mlp)
    
    # Language model requires handling both model and lm_head
    print("train llm: ", model_args.tune_mm_llm)
    set_trainable_status(model.model, model_args.tune_mm_llm)
    model.lm_head.requires_grad = model_args.tune_mm_llm

    # Handle panoramic adapter modules
    print(f"{'Unfreezing' if model_args.tune_mm_adaptorformer else 'Freezing'} the panoramic adapter modules.")
    for name, param in model.visual.named_parameters():
        if "panoramic" in name:
            param.requires_grad = model_args.tune_mm_adaptorformer

    # --- Parameter status report ---
    calculate_parameters_zero3(model, "Whole Model")
    calculate_parameters_zero3(model.visual, "Visual Tower")
    calculate_parameters_zero3(model.visual.merger, "Vision-Language Merger")
    calculate_parameters_zero3(model.model, "Language Model")
    
    calculate_adapter_parameters_zero3(model, adapter_keyword="panoramic")

def calculate_adapter_parameters_zero3(model, adapter_keyword="panoramic"):
    """Calculate adapter parameters for DeepSpeed ZeRO-3."""
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    
    # Get base model
    if isinstance(model, DeepSpeedEngine):
        base_model = model.module
    else:
        base_model = model
    
    if not hasattr(base_model, 'visual'):
        print(f"\n--- {adapter_keyword.capitalize()} Adapter Parameter Status ---")
        print("WARNING: No 'visual' module found in model!")
        print("----------------------------\n")
        return
    
    total_adapter_params = 0
    trainable_adapter_params = 0
    
    for n, p in base_model.visual.named_parameters(recurse=True):
        if adapter_keyword not in n:
            continue
        
        # ZeRO-3 requires temporary parameter gathering
        if hasattr(p, 'ds_status') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                param_size = p.numel()
        else:
            param_size = p.numel()
        
        total_adapter_params += param_size
        if p.requires_grad:
            trainable_adapter_params += param_size
    
    print(f"\n--- {adapter_keyword.capitalize()} Adapter Parameter Status (ZeRO-3) ---")
    print(f"Total Parameters:      {total_adapter_params / 1e6:.2f} M")
    print(f"Trainable Parameters:  {trainable_adapter_params / 1e6:.2f} M")
    
    if total_adapter_params > 0:
        percentage = (trainable_adapter_params / total_adapter_params) * 100
        print(f"Trainable Percentage:  {percentage:.2f}%")
    else:
        print("Trainable Percentage:  N/A (0 total parameters)")
    print("----------------------------\n")

def calculate_parameters_zero3(module: nn.Module, name: str):
    """Calculate parameters for DeepSpeed ZeRO-3."""
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    
    total_params = 0
    trainable_params = 0
    
    for param in module.parameters():
        # ZeRO-3 requires temporary parameter gathering
        if hasattr(param, 'ds_status') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            with deepspeed.zero.GatheredParameters([param]):
                param_size = param.numel()
        else:
            param_size = param.numel()
        
        total_params += param_size
        if param.requires_grad:
            trainable_params += param_size
    
    print(f"\n--- {name} Parameter Status (ZeRO-3) ---")
    print(f"Total Parameters:      {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters:  {trainable_params / 1e6:.2f} M")
    
    if total_params > 0:
        percentage = (trainable_params / total_params) * 100
        print(f"Trainable Percentage:  {percentage:.2f}%")
    print("----------------------------")

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, AdaptorArguments)
    )
    model_args, data_args, training_args, adaptor_args = parser.parse_args_into_dataclasses()
    
    setup_logger(training_args.logging_dir)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path) 

    adaptor_config = None
    try:
        with open(os.path.join(model_args.model_name_or_path, "adaptor_config.json"), "r") as f:
            adaptor_config = json.load(f)
        print(f"Loaded adaptor_config.json from {model_args.model_name_or_path}")
    except:
        parent_dir = os.path.dirname(model_args.model_name_or_path)
        try:
            with open(os.path.join(parent_dir, "adaptor_config.json"), "r") as f:
                adaptor_config = json.load(f)
            print(f"Loaded adaptor_config.json from {parent_dir}")
        except:
            print("No existing adaptor_config.json found, using command line arguments.")
            
    if adaptor_config is None:
        print("Using adaptor arguments from command line.")
        adaptor_config = vars(adaptor_args).copy()

    with open(os.path.join(training_args.output_dir, "adaptor_config.json"), 'w') as json_file:
        json.dump(adaptor_config, json_file, indent=4)
        
    modify_qwen2_5_vl_vision_attention(adaptor_args.adaptor_name, config.vision_config, adaptor_config)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            # ignore_mismatched_sizes=True
            # torch_dtype=torch.half,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"

    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            # torch_dtype=torch.half,
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)
    # training_args.safe_serialization = False

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if torch.distributed.get_rank() == 0:
        print("model trainable parameters:")
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

