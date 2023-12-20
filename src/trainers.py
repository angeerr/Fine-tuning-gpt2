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
from tqdm import tqdm
from utils import *
#class DPOTrainer:

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

        '''Initialization'''
        self.run_name = 'SGD_Mom_lora1' # perfix of file
        save_step = 10000
        eval_step = 200
        test_num = 240
        
        train_data = self.train_dataloader 
        test_data = self.test_dataloader
        model = self.model.to(self.device)
        
        #optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),weight_decay=1e-6) #AdamW
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-1) #SGD
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=1e-6) #SGD with momentum
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9, nesterov=True, weight_decay=1e-1) #SGD with Nesterov

        self.optimizer = optimizer
        lossList = []

        allocated_before, cached_before = get_gpu_memory()
        #allocated_before = get_gpu_memory()
        for i in tqdm(range(1, self.cfg.max_steps+1), desc='Step', leave=False): #
            '''training'''
            x_train, y_train = next(train_data)
            x_train = x_train.to(self.device) 
            y_train = y_train.to(self.device) 
            _ , train_loss = model(x = x_train, targets = y_train)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            '''evaluation'''
            if i % eval_step == 0:
                model.eval()
                
                with torch.no_grad():
                    loss_ = 0
                    for _ in tqdm(range( int(test_num//self.cfg.batch_size)), desc='Step', leave=False): 
                        x_test, y_test = next(test_data)
                        _ , test_loss = model(x=x_test.to(self.device), targets=y_test.to(self.device))
                        loss_ += test_loss.item()
                    model.train()
                
                ### Record train & test loss
                lossList.append({"iter":i, "train loss": train_loss.item(), "test loss":loss_/(int(test_num//self.cfg.batch_size))})
                self.save_metrics(lossList)
            
            else:
                ### Record train loss
                lossList.append({"iter":i, "train loss":train_loss.item()})
            
            '''Save model as checkpoints'''
            if i % save_step == 0:
                self.save_states(i, i==self.cfg.max_steps)
        
        allocated_after, cached_after = get_gpu_memory()
        #allocated_after = get_gpu_memory()

        logging.info(f"Memory Allocated: {allocated_after - allocated_before} bytes")
        logging.info(f"Memory Cached: {cached_after - cached_before} bytes")

        return lossList
