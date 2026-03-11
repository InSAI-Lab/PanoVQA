# [CVPR 2026]: More than the Sum: Panorama-Language Models for Adverse Omni-Scenes 

![PanoVQA](fig1.png "Overview of Panorama-Language Modeling (PLM).")


**Paper Link:** [Coming Soon]()

## 📊 Datasets

Download the datasets from Google Drive:
- **PanoVQA:** [Download Link](https://drive.google.com/drive/folders/1NOpXK-oR6P4JEm4ewuwkF29xV3kS-zE4?usp=drive_link)
- **PanoVQA-mini:** [Download Link](https://drive.google.com/drive/folders/1jtoEJtUBpen3OS4G_udl2zODKSKYKT4m?usp=drive_link)

### Structure
Please organize your workspace as follows:
```text
Workspace/
├── PanoVQA/
│   ├── BlendPASS/
│   ├── DeepAccident/
│   └── NuScenes/
├── PanoVQA_mini/
│   ├── BlendPASS/
│   ├── DeepAccident/
│   └── NuScenes/
└── Panorama/
    └── images/
```

## 🚀 Training

Navigate to the fine-tuning directory and run your training script:
```bash
cd plm-finetune
sh scripts/sft_3b.sh  # or your respective script
```

## ⚖️ Evaluation

### Inference Phase
Run inference to generate predictions:
```bash
cd eval_benchmark
python eval.script.mini.adaptor.py \
  --model_path "../plm-finetune/output/adaptor/Pano_adaptor_former_finetune(adapter,llm,mlp)" \
  --save_path "outputs/inferences/sft3b.json"
```

### LLM-as-a-Judge Phase
Evaluate the generated predictions using an OpenAI API key:
```bash
python get_gpt_score.py \
  --input outputs/inferences/sft3b.json \
  --output outputs/gpt_score/sft3b.json \
  -k "YOUR_OPENAI_API_KEY"
```

## 🎯 Citation
```bibtex
```

## 🙏 Acknowledgments
This work is based on the [Qwen-vl](https://github.com/QwenLM/Qwen3-VL) repository. Huge thanks to their contributors!
