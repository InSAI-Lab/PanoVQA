# [CVPR 2026]: More than the Sum: Panorama-Language Models for Adverse Omni-Scenes 

![PanoVQA](assets/fig1.png "Overview of Panorama-Language Modeling (PLM).")

Demo:
Comming Soon~

**Paper Links:** 
<!-- [Coming Soon](https://arxiv.org/abs/2603.09573) -->

[![arXiv](https://img.shields.io/badge/arXiv-2511.14097-b31b1b.svg)](https://arxiv.org/abs/2603.09573)
[![PDF](https://img.shields.io/badge/PDF-Download-red.svg)](https://arxiv.org/pdf/2603.09573)
## 📊 Datasets

Download the datasets from Google Drive:
- **PanoVQA:** [Download Link](https://drive.google.com/drive/folders/1NOpXK-oR6P4JEm4ewuwkF29xV3kS-zE4?usp=drive_link)
- **PanoVQA-mini:** [Download Link](https://drive.google.com/drive/folders/1jtoEJtUBpen3OS4G_udl2zODKSKYKT4m?usp=drive_link)

and unzip them.

### Structure
Please organize your workspace as follows. Ensure all datasets are placed under the corresponding directories:
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

## 🛠️ Environment Setup

We recommend installing the required dependencies with the following commands to match the Qwen-VL ecosystem:
```bash
pip install transformers==4.51.3 accelerate
```

We offer a toolkit to help you handle various types of visual input more conveniently. We highly recommend using the `[decord]` feature for faster video loading:
```bash
pip install qwen-vl-utils[decord]
```

**Flash-Attention 2 to speed up generation**  
For better acceleration and memory savings, especially in multi-image and video scenarios, make sure to install the latest version of Flash Attention 2:
```bash
pip install -U flash-attn --no-build-isolation
```

## 🚀 Training
Configure your data at:
```bash
./PLM/plm-finetune/plm/data/__init__.py
```


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
ArXiv version:
```bibtex
@article{fan2026PanoVQA,
  title={More than the Sum: Panorama-Language Models for Adverse Omni-Scenes},
  author={Fan, Weijia and Liu, Ruiping and Wei, Jiale and Chen, Yufan and Zheng, Junwei and Zeng, Zichao and Zhang, Jiaming and Li, Qiufu and Shen, Linlin and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2603.09573},
  year={2026}
}
```

CVPR version:
```bibtex
@article{fan2026PanoVQA,
  title={More than the Sum: Panorama-Language Models for Adverse Omni-Scenes},
  author={Fan, Weijia and Liu, Ruiping and Wei, Jiale and Chen, Yufan and Zheng, Junwei and Zeng, Zichao and Zhang, Jiaming and Li, Qiufu and Shen, Linlin and Stiefelhagen, Rainer},
  booktitle={2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
## 🙏 Acknowledgments
This work was supported by the Shenzhen University Overseas Exchange Scholarship, which supported my living expenses in Karlsruhe. I had a very nice time in Karlsruhe.

This work is based on the [Qwen-VL](https://github.com/QwenLM/Qwen3-VL) team repository. Huge thanks to their contributors and the community!
