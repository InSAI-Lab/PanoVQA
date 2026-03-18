python eval.script.mini.adaptor.py --model_path "/home/hk-project-pai00053/wd1434/programmes/PLM/plm-finetune/output/plm/plm3b_psa(adapter,llm,mlp)" --save_path "outputs/inferences/plm3b_psa_mini.json"

python get_gpt_score.py --input outputs/inferences/plm3b_mini.json --output outputs/gpt_score/plm3b_mini.json


python get_gpt_score.py --input outputs/inferences/plm3b_mini.json --output outputs/gpt_score/plm3b_mini.json
