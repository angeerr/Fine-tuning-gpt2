import click
import torch
from trainers import SFTTrainer
from gpt import GPT
from dataset import EYLSFTStaticDataset
from configs import get_configs
import logging
from utils import *

# Avoid GPU version conflict (For Kaggle GPU only). Comment below two lines if you use local machine in order to speed up training.
import torch._dynamo.config
torch._dynamo.config.suppress_errors = True


def train(pretrain, batch_size, exp_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = get_configs("gpt2-medium/lora") # change this line to select different models #gpt2-medium/lora
    cfg.max_steps = 160000 // batch_size
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    #assert pretrain == "huggingface" # make sure the pretrained model is in the format of huggingface.
    cfg.exp_name = exp_name

    # load the pretrained GPT model based on the configuration
    model = GPT.from_pretrained(cfg)
    
    # load SFT dataset
    train_ds = EYLSFTStaticDataset(block_size=1024,
                                   split='train',
                                   max_examples=None,
                                   tokenizer_name="tiktoken/gpt2")
    #train_size = int(0.8 * train_ds.__len__())  # use 80% data in train_dataset as train dataset

    # compute the size of validation set
    #val_size = train_ds.__len__() - train_size  # use 20% data in train_dataset as validation dataset
    #print(train_size, val_size, train_ds.__len__())
    # use random_split to split train dataset and validation dataset
    #train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    test_ds = EYLSFTStaticDataset(block_size=1024,
                                  split='test',
                                  max_examples=None,
                                  tokenizer_name="tiktoken/gpt2")
    
    trainer = SFTTrainer(cfg, device, model, train_ds, test_ds)
    lossList = trainer.fit()
    return lossList


@click.command()
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=8)
@click.option('--exp-name', '-n', default="default")
def main(pretrain, batch_size, exp_name):
    torch.manual_seed(1234)
    
    
    lossList = train(pretrain, batch_size, exp_name)
    

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("No GPUs available.")

    
    
    draw_loss(lossList, 'train', '/mntcephfs/lab_data/mazhuoheng/MDS5210-23fall/src/runs/log/SGD_Mom_lora1')
    draw_loss(lossList, 'test', '/mntcephfs/lab_data/mazhuoheng/MDS5210-23fall/src/runs/log/SGD_Mom_lora1')


if __name__ == "__main__":
    logging.basicConfig(filename="/mntcephfs/lab_data/mazhuoheng/MDS5210-23fall/src/runs/log/SGD_Mom_lora1/SGD_Mom_lora1.log", level=logging.INFO)
    main()
