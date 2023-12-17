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
- Run prepare_sft_dataset.py to generate two files: sft_train.json & sft_test.json
- Train gpt2-medium
  | Detail | Value |
  | ---- | ---- |
  | Model | gpt2-medium |
  | Train iteration | 160000 |
  | Batch size | 8 |
  | Optimizer | AdamW (weight decay) |
  | Test step | every 200 steps |
  | Other Hyper-parameter | <link> |
  | Training Record | <link> |
  | Train error | <link> |
  | Test error | <link> |
❗ 
