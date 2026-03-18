import torch
from torch import nn
import os
import json
from transformers import AutoModel, AutoTokenizer, Qwen2_5_VLForConditionalGeneration,Qwen2VLImageProcessor 
import torch.nn.functional as F

import torch
from torch import nn
from typing import Optional, Tuple
import copy
import sys
import math
import torchvision
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import os
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionBlock, Qwen2_5_VisionTransformerPretrainedModel,apply_rotary_pos_emb_vision
from qwen_vl_utils import *
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoConfig, AutoProcessor



# Qwen2_5_VisionTransformerPretrainedModel
class Qwen2_5_VisionTransformerPretrainedModel_Pano(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        print("load Qwen2_5_VisionTransformerPretrainedModel_Pano")
        
        # self.fullatt_block_indexes
        # self.blocks = nn.ModuleList(
        #     [Qwen2_5_VLVisionBlock_Pano(config, config._attn_implementation) for _ in range(config.depth)]
        # )
    def _calculate_patch_coords_from_grid_thw(self, grid_thw: torch.Tensor) -> torch.Tensor:
            """
            根据 grid_thw (T, H_patches, W_patches) 为批处理中的所有 token 生成 (y, x) 坐标。
            
            参数:
                grid_thw (torch.Tensor): 形状 [num_images, 3]，
                                        每一行是 [T, H_patches, W_patches]
            
            返回:
                torch.Tensor: 形状 [Total_Seq_Len, 2]，存储 (y, x) 坐标
            """
            all_patch_coords = []
            B = grid_thw.shape[0] # B = num_images
            device = grid_thw.device
            
            # 遍历批处理中的每张图像
            for i in range(B):
                # 提取 T, H_patches, W_patches
                T_int = int(grid_thw[i, 0])
                H_patches_int = int(grid_thw[i, 1])
                W_patches_int = int(grid_thw[i, 2]) # 这就是你关心的 600

                # 1. 为 (H, W) 网格创建 1D 坐标
                # y_coords: [0, ..., H_p-1]
                y_coords = torch.arange(H_patches_int, device=device, dtype=torch.float32)
                # x_coords: [0, ..., W_p-1] (例如 [0, ..., 599])
                x_coords = torch.arange(W_patches_int, device=device, dtype=torch.float32)
                
                # 2. 创建 2D 网格: grid_y, grid_x 形状都是 [H_p, W_p]
                grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
                
                # 3. 堆叠为 (y, x) 对: [H_p, W_p, 2]
                coords_2d = torch.stack([grid_y, grid_x], dim=-1)
                
                # 4. 展平为 [N, 2], N = H_p * W_p
                coords_2d_flat = coords_2d.reshape(-1, 2)
                
                # 5. 为 T (时序) 维度平铺
                # (T, N, 2) -> [T*N, 2]
                # 这与 cu_seqlens 的计算方式 (repeat_interleave) 保持一致
                coords_final_for_img_i = coords_2d_flat.repeat(T_int, 1) 
                
                all_patch_coords.append(coords_final_for_img_i)
            
            # 6. 将 batch 中所有图像的坐标拼接在一起
            # 形状: [Total_Seq_Len, 2]
            # Total_Seq_Len 正好等于 hidden_states.shape[0]
            patch_coords_full = torch.cat(all_patch_coords, dim=0)
            
            return patch_coords_full
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # grid_thw = image_grid_thw
        hidden_states = self.patch_embed(hidden_states) # unifity dimenion, [B image token, dim]
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw) # 这边已经是windows attention了
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        # self._calculate_patch_coords_from_grid_thw()
        patch_coords = self._calculate_patch_coords_from_grid_thw(grid_thw)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        # original_cu_seqlens = cu_seqlens
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes: # full attention.
                cu_seqlens_now = cu_seqlens
                full_attention = True
            else: # windows attention
                cu_seqlens_now = cu_window_seqlens
                full_attention = False
            hidden_states = blk(hidden_states, patch_coords=patch_coords,cu_seqlens=cu_seqlens_now, cu_seqlens_whole=cu_seqlens, position_embeddings=position_embeddings, grid_thw=grid_thw, full_attention=full_attention)
            # 
            # if self.gradient_checkpointing and self.training:
            #     hidden_states = self._gradient_checkpointing_func(
            #         blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings,
            #         # grid_thw=grid_thw, 
            #         # use_reentrant=False
            #     )
            # else:
            #     hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, grid_thw=grid_thw)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states
# Your modified Vision Block, now more flexible
class Qwen2_5_VLVisionBlock_Pano(Qwen2_5_VLVisionBlock):
    def __init__(self, config, atten_adapter: nn.Module = None, FFN_adapter: nn.Module = None, attn_implementation: str = "sdpa") -> None:
        super().__init__(config, attn_implementation)
        self.panoramic_attention = atten_adapter if atten_adapter is not None else None
        self.panoramic_adapter = FFN_adapter if FFN_adapter is not None else None
        
    def _init_weights(self, module):
        # 在权重初始化函数中添加
        if isinstance(module, nn.BatchNorm2d):
            # 确保 BatchNorm 的 running_mean 和 running_var 在正确设备上
            module.running_mean = module.running_mean.cuda() if torch.cuda.is_available() else module.running_mean
            module.running_var = module.running_var.cuda() if torch.cuda.is_available() else module.running_var

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        # You must pass the feature map's H and W to the forward pass
        # This needs to be determined from your model's configuration
        patch_coords: torch.Tensor,         # <--- **关键新增**：必须传入 patch 坐标
        cu_seqlens_whole: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        grid_thw=None,
        full_attention=False
    ) -> torch.Tensor:
        # instruction:
        # hidden_states: 输入的张量，形状是 (总令牌数, dim)。注意，这里没有显式的 batch 维度。所有图像的令牌都被拼接成一个长序列。
        # cu_seqlens: 这是理解此模块的关键。它的全称是 cumulative sequence lengths（累积序列长度）。它是一个一维张量，记录了每个图像在 hidden_states 中结束的位置。
            # 例如，如果 cu_seqlens 是 [0, 196, 400]，这意味着：
            # 第一个图像的令牌范围是 hidden_states[0:196]（长度196）。
            # 第二个图像的令牌范围是 hidden_states[196:400]（长度204）。
        # position_embeddings: 预先计算好的 RoPE 的 cos 和 sin 值。这是目前推荐的方式。
        # rotary_pos_emb: 旧版的 RoPE 参数，未来将被移除。

        # Standard transformer block operations
        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        
        hidden_states = self.attn(
            norm_hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        
        if self.panoramic_attention and (not full_attention):
            pano_hidden_states = self.panoramic_attention(
                hidden_states=norm_hidden_states,
                patch_coords=patch_coords,
                rotary_pos_emb=rotary_pos_emb,
                cu_seqlens=cu_seqlens,
                cu_seqlens_whole=cu_seqlens_whole,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + hidden_states + pano_hidden_states
        else:
            hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        
        # Main MLP
        mlp_output = self.mlp(hidden_states)
        
        # Apply the panoramic adapter
        if self.panoramic_adapter:
            adapter_output = self.panoramic_adapter(hidden_states, grid_thw)
        
            hidden_states = residual + mlp_output + adapter_output
        else:
            hidden_states = residual + mlp_output
        
        return hidden_states
