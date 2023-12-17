# MDS5210-23fall


To do:
-
- report


finish:
-
- trainer.py train a simple model @jiangan
- rest @zhuoheng



Process:
-
4.1 (A)
- Run prepare_sft_dataset.py to generate two files: sft_train.json & sft_test.json
- Train gpt2-medium
  | Detail | Value |
  | ---- | ---- |
  | Model | gpt2-medium |
  | Train iteration | 160000 |
  | Batch size | 8 |
  | Optimizer | AdamW (weight decay) |
  | Test step | 200 steps |
  | Test data | 240 |
  | Other Hyper-parameter | <link> |
  | Training Record | <link> |
  | Train error | <link> |
  | Test error | <link> |
❗ Train 20000 (Train iteration / batch size) steps, and test in every 200 steps

  
4.2(B)
- Run eval.py to evaluate the performance of vanilla gpt2-medium and sft gpt2-medium
  
❗ eval.py leverages the reward model (OpenAssistant/rewardmodel-deberta-v3-large-v2) to evaluate the performance instead of using Openai Apikey  

🎨 eval.py is modified based on evaluate.py  

😊 [Reward model]<https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2> provides you a quick start of reward-model-deberta-v3-large-v2.
