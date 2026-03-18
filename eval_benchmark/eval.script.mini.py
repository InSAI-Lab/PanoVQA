import argparse
import os, sys, json
from tqdm import tqdm
import torch

sys.path.append("/home/hk-project-pai00053/wd1434/programmes/PLM")
sys.path.append("/home/hk-project-pai00053/wd1434/programmes/PLM/plm-finetune")
sys.path.append("/home/hk-project-pai00053/wd1434/programmes/PLM/plm-utils/src")

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from plm.data import val_pano_mini_data_dict
from qwen_vl_utils import process_vision_info


val_pano_data_dict = val_pano_mini_data_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for Qwen2.5-VL on val_data_dict.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model directory')
    parser.add_argument('--save_path', type=str, required=False, default=None, help='Path to save result json')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--min_pixels', type=int, default=256*28*28)
    parser.add_argument('--max_pixels', type=int, default=1280*28*28)
    parser.add_argument('--resize_width', type=int, default=9600//3)
    parser.add_argument('--resize_height', type=int, default=600//3)
    return parser.parse_args()

def get_conversation_pair(conv):
    # Support both formats
    if isinstance(conv[0], dict) and 'content' in conv[0]:
        # Format 1
        user = conv[0]['content'] if 'content' in conv[0] else conv[0].get('value', '')
        gt = conv[1]['content'] if 'content' in conv[1] else conv[1].get('value', '')
    elif isinstance(conv[0], dict) and 'value' in conv[0]:
        # Format 2
        user = conv[0]['value']
        gt = conv[1]['value']
    else:
        user = str(conv[0])
        gt = str(conv[1])
    return user, gt

def main():
    args = parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    )
    # try:
    #     processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    # except:
    processor = AutoProcessor.from_pretrained("/home/hk-project-pai00053/wd1434/workspace/huggingface/Qwen2.5-VL-3B-Instruct", min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    processor.tokenizer.padding_side = "left"

    # Checkpoint/resume support
    if args.save_path is None:
        args.save_path = os.path.join(args.model_path, "eval_results.json")

    # Flat result dict: id -> {...}
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for dataset_name, dataset_info in val_pano_data_dict.items():
        print(f"\n=== Evaluating {dataset_name} ===")
        ann_path = dataset_info["annotation_path"]
        if not os.path.exists(ann_path):
            print(f"Annotation file not found: {ann_path}")
            continue
        with open(ann_path, 'r') as f:
            data = json.load(f)
        id_set = set(all_results.keys())
        batch_size = args.batch_size
        num_samples = len(data)
        with torch.no_grad():
            for start_idx in tqdm(range(0, num_samples, batch_size), desc=f"{dataset_name}"):
                end_idx = min(start_idx + batch_size, num_samples)
                batch = data[start_idx:end_idx]
                # Skip if all in batch are already done
                if all(str(item['id']) in id_set for item in batch):
                    continue
                batch_ids = []
                batch_questions = []
                batch_gt_answers = []
                batch_categories = []
                batch_messages = []
                for val_json in batch:
                    if str(val_json['id']) in id_set:
                        continue
                    batch_ids.append(val_json['id'])
                    user, gt = get_conversation_pair(val_json['conversations'])
                    batch_questions.append(user)
                    batch_gt_answers.append(gt)
                    batch_categories.append(val_json['id'].split('_')[0])
                    if 'image' in val_json:
                        image_path = os.path.relpath(os.path.join(os.path.dirname(ann_path), val_json['image']), os.getcwd())
                    else:
                        image_path = None
                    if '<image>' in user:
                        content_list = []
                        for line in user.split('\n'):
                            if '<image>' in line:
                                cam = line.split(':')[0] if ':' in line else 'image'
                                content_list.append({"type": "image", "image": image_path or '', "name": cam, "resized_height": args.resize_height, "resized_width": args.resize_width})
                            elif line.strip():
                                content_list.append({"type": "text", "text": line.strip()})
                        message = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": content_list}
                        ]
                    else:
                        message = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image_path or '', "resized_height": args.resize_height, "resized_width": args.resize_width},
                                    {"type": "text", "text": user}
                                ]
                            }
                        ]
                    batch_messages.append(message)

                if not batch_ids:
                    continue

                batch_texts = [
                    processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                    for message in batch_messages
                ]

                batch_image_inputs = []
                batch_video_inputs = []
                for message in batch_messages:
                    image_inputs, video_inputs = process_vision_info(message)
                    batch_image_inputs.append(image_inputs)
                    batch_video_inputs.append(video_inputs)

                images = batch_image_inputs if any(batch_image_inputs) else None
                videos = batch_video_inputs if any(batch_video_inputs) else None

                inputs = processor(
                    text=batch_texts,
                    images=images,
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for i in range(len(batch_ids)):
                    all_results[str(batch_ids[i])] = {
                        "question": batch_questions[i],
                        "gt_answer": batch_gt_answers[i],
                        "pred_answer": output_texts[i],
                        "category": batch_categories[i]
                    }
                # Save after each batch for checkpointing
                with open(args.save_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

    print(f"\nAll datasets finished. Results saved to {args.save_path}")

if __name__ == "__main__":
    main()
