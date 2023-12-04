import torch
import os
import json
import random
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from configs import TrainingConfig
import logging
from evaluate import *

class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')
    
class SFTTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        torch.cuda.empty_cache()
        self.cfg = cfg
        self.device = device
        self.run_name = f"sft_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"

        # create a dataloader
        # get a batch of (data, label) by: x, y = self.train_dataloader
        self.train_dataloader = iter(
            DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))

        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.model = model
        self.dtype = torch.float16

        self.finetune_method = cfg.finetune_method

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        # TODO: complete the SFT training.
        train_data = self.train_dataloader # TODO: How to add data
        test_data=self.test_dataloader
        model=self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-1)
        test_losses=[]
        train_losses=[]
        losses={}
        for iter in range(self.cfg.total_epochs):

            for i, data in enumerate(train_data):
                    #logging.info(f"iter: {iter}, train loss {losses['train']:.4f}, val {losses['val']:.4f}")
                optimizer.zero_grad(set_to_none=True)
                x_train, y_train = data
                x_train=x_train.to(self.device)
                y_train = y_train.to(self.device)
                logits, loss = model.forward(x=x_train, targets=y_train)
                if i % 100 == 0:
                #or i == self.cfg.max_steps - 1:
                    #logits, loss = model.forward(x=x_train,targets=y_train)
                #loss=loss.to(self.device)
                    #with torch.no_grad():
                        #for j, tes_data in enumerate(test_data):
                            #x_test, y_test = tes_data
                            #x_test=x_test.to(self.device)
                            #y_test=y_test.to(self.device)
                            #logits, test_loss = self.model.forward(x=x_test, targets=y_test)
                            #total_test_loss += test_loss
                    #losses['train'] =loss # TODO: Implementation of function
                    #losses['val']= total_test_loss/len(test_data)
                    train_losses.append(loss)
                    #test_losses.append(losses['val'])
                    #print("iter={}, train loss={}, test loss={}".format(iter,losses['train'],losses['val']))
                    print("iter={}, train loss={}".format(i,loss))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
