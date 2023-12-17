# MDS5210-23fall


To do:
-
- report


finish:
-
- trainer.py train a simple model @jiangan
- rest @zhuoheng

Note:
-
⚠ Since the code is runned on the computation node (single GPU version) on clusters, thus, some codes are ignorable when being runned on your server or kaggle

Process:
-
**Base Tasks**
*4.1 (A)*
- Run `prepare_sft_dataset.py` to generate two files: sft_train.json & sft_test.json
- Train gpt2-medium
  | Setting | Value |
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

  
*4.1(B)*
- Run `eval.py` to evaluate the performance of vanilla gpt2-medium and sft gpt2-medium, [result screenshot]<link>, [result detail]<link>
  
🚀 `eval.py` (modified on `evaluate.py`) leverages the reward model [OpenAssistant/rewardmodel-deberta-v3-large-v2](<https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2>) to evaluate the performance instead of using Openai Apikey  

4.1(c)
- Insights: Summarize what you find based on the results or settings from 4.1(A) and 4.1(B).

**Explorations**
*4.2(c)*
- Run `train_sft.py`. The code of lora has been integrated by TA, only change `cfg = get_configs("gpt2-medium")` to `cfg = get_configs("gpt2-medium/lora")` in train_sft.py to add lora on gpt2-medium
- Comparsion
  | Setting | Value |
  | ---- | ---- |
  | gpt2-medium train error | <link> |
  | gpt2-medium/lora train error | <link> |

❗ `gpt2-medium` and `gpt2-medium/lora` are trained based on same hyper-parameter settings, optimizer: AdamW (weight decay)
  | Lora Rank | Value |
  | ---- | ---- |
  | lora rank = 1 | <link> |
  | lora rank = 10 | <link> |
  | lora rank = 100 | <link> |
❗ Different lora rank training are based on same hyper-parameter settings, optimizer: AdamW (weight decay)

*4.2(a)*
- Run `train_sft.py`. Only need to switch the optimizer from AdamW to others (already included in `fit()` function in `trainers.py`) and then test on different optimizers on gpt2-medium/lora (save time)
- Comparsion
  | Optimizers | Value |
  | ---- | ---- |
  | SGD | <link> |
  | SGD with Momentum (momentum=0.9) | <link> |
  | SGD with Nesterov (momentum=0.9)| <link> |
  | AdamW ($\beta_1=0.9$, $\beta_2=0.95$) | <link> |
❗ Models with different optimizers are trained with the same weight decay and hyper-parameter settings
  
  
