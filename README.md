# MDS5210-23fall

Note:
-
⚠ 
- Since the code is runned on the computation node (single GPU version) on clusters, thus, some codes are ignorable when being runned on your server or kaggle
- Some model paths should be modified according to your local environments

Process:
-
**Base Tasks**  

*4.1 (A)*
- Run `prepare_sft_dataset.py` to generate two files: `sft_train.json` and `sft_test.json`
- Train gpt2-medium
  | Setting | Value |
  | ---- | ---- |
  | Model | gpt2-medium |
  | Train iteration | 160000 |
  | Batch size | 8 |
  | Optimizer | AdamW (weight decay) |
  | Test step | 200 steps |
  | Test data | 240 |
  | Other Hyper-parameter | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW/hyperparams.json>) |
  | Training Record | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW/metrics.json>) |
  | Train error | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW/train.jpg>) |
  | Test error | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW/test.jpg>) |
❗ Train 20000 (Train iteration / batch size) steps, and test in every 200 steps

  
*4.1(B)*
- Run `eval.py` to evaluate the performance of vanilla gpt2-medium and sft gpt2-medium, [result detail](<https://github.com/roy-mzh/MDS5210-23fall/tree/main/src/eval_result>)
  
🚀 `eval.py` (modified on `evaluate.py`) leverages the reward model [OpenAssistant/rewardmodel-deberta-v3-large-v2](<https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2>) to evaluate the performance instead of using Openai Apikey  

*4.1(C)*
- Insights: Summarize what you find based on the results or settings from 4.1(A) and 4.1(B).

**Explorations**  
  
*4.2(C)*
- Run `train_sft.py`. The code of lora has been integrated by TA, only change `cfg = get_configs("gpt2-medium")` to `cfg = get_configs("gpt2-medium/lora")` in train_sft.py to add lora on gpt2-medium
- Comparsion
  | Setting | Figure Link |
  | ---- | ---- |
  | gpt2-medium train error | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW/train.jpg>) |
  | gpt2-medium test error | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW/test.jpg>) |
  | gpt2-medium/lora train error | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW_lora1/train.jpg>) |
  | gpt2-medium/lora test error | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW_lora1/test.jpg>) |

❗ `gpt2-medium` and `gpt2-medium/lora` are trained based on same hyper-parameter settings, optimizer: AdamW (weight decay)
  | Lora Rank | Figure Link | Dialogue Quality |
  | ---- | ---- | ---- |
  | Full paramters | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW/>) | $0.85$ |
  | lora rank = 1 | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW_lora1/> )| $0.51$ |
  | lora rank = 10 | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW_lora10/>) | $0.45$ |
  | lora rank = 100 | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW_lora100/>) | $0.49$ |

❗ Different lora rank training are based on same hyper-parameter settings, optimizer: AdamW (weight decay)

*4.2(A)*
- Run `train_sft.py`. Only need to switch the optimizer from AdamW to others (already included in `fit()` function in `trainers.py`) and then test on different optimizers
- Comparsion
  | Optimizers | Figure Link | GPU Memory |
  | ---- | ---- | ---- |
  | SGD | [Here](<https://github.com/roy-mzh/MDS5210-23fall/tree/main/src/runs/SGD_lora1>) | $1663992832$ bytes |
  | SGD with Momentum (momentum=0.9) | [Here](<https://github.com/roy-mzh/MDS5210-23fall/tree/main/src/runs/SGD_Mom_lora1>) | $1877104128$ bytes |
  | SGD with Nesterov (momentum=0.9) | [Here](<https://github.com/roy-mzh/MDS5210-23fall/tree/main/src/runs/SGD_Nest_lora1>) | $1877104128$ bytes |
  | AdamW ($\beta_1=0.9$, $\beta_2=0.95$) | [Here](<https://github.com/roy-mzh/MDS5210-23fall/blob/main/src/runs/AdamW/>) | $2090215424$ bytes |
  
❗ Models with different optimizers are trained with the same weight decay and hyper-parameter settings

❗ 4.2(A) experiments are conducted based on gpt2-medium/lora with rank $1$ (save time). In the report this should be specified explitcitly
  
  
