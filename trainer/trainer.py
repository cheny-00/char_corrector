
import os
from attr import has
import torch
from tqdm import tqdm
from time import time
from functools import partial


class BaseTrainer:
    
    def __init__(self,
                 optim,
                 seq_len,
                 batch_size,
                 emb_dim,
                 train_data,
                 dev_data,
                 tb_writter,
                 logger,
                 log_interval,
                 eval_interval,
                 scheduler=None,
                 ckp_save_path=None,
                 train_step=0,
                 device='cpu'):
        
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        self.train_data = train_data
        self.dev_data = dev_data
        
        self.optim = optim
        self.scheduler = scheduler
        
        self.ckp_save_path = ckp_save_path
        self.tb_writter = tb_writter
        self.logger = logger
        
        self.train_step = train_step
        self.hidden = None
        self.device = device
        
    
    
    def train(self,
              model,
              epochs,
              run_process,
              criterion):
        


        log_interval, eval_interval = self.log_interval, self.eval_interval

        self.device = next(model.parameters()).device
        
        model.train()
        best_score = float('inf')
        train_loss = 0
        for i in range(epochs):
            tqdm_train_iter = tqdm(enumerate(self.train_data), desc="train", total=len(self.train_data))
            log_start_time = time()

            for num_batch, data in tqdm_train_iter:
                
                loss = run_process(data=data, )

                train_loss += loss
                # TODO will we use fp16?
                self.train_step += 1
                # nn.utils.clip_grad_norm(model.parameters(), 10.0) #TODO clip?

                if  self.train_step % log_interval == 0:

                    cur_loss = train_loss / log_interval
                    elapsed = time() - log_start_time
                    cur_lr = self.optim.param_groups[0]['lr']
                    
                    if self.tb_writter is not None:
                        self.tb_writter.add_scalar('loss', cur_loss, self.train_step)
                        self.tb_writter.add_scalar('lr', cur_lr, self.train_step)
                        self.tb_writter.add_scalar('elapsed', elapsed, self.train_step)
                    
                    desc_txt =  f"| Epoch: {i} | steps: {self.train_step}| {num_batch} batches | loss: {cur_loss: 5.6f} | lr: {cur_lr: 5.6f} | elapsed: {elapsed}"

                    tqdm_train_iter.set_description(desc=desc_txt, refresh=True)
                    train_loss = 0
                    log_start_time = time()
            num_batch=0
            if (i + 1) % eval_interval == 0:

                dev_start_time = time()
                scores = self.eval_model(model, criterion=criterion) # TODO should we add early stop?
                model.train()
                dev_elapsed = time() - dev_start_time
                
                valid_score = scores['valid_score']
                dev_desc_txt =  f"| Dev at epoch: {i} | steps: {self.train_step}| {num_batch} batches | valid_score: {valid_score: 5.6f} | best_score: {best_score: 5.6f}| elapsed: {dev_elapsed} |"
                if self.tb_writter is not None:
                    self.tb_writter.add_scalar('dev_score', valid_score, self.train_step)
                self.logger('-' * len(dev_desc_txt) + "\n" + dev_desc_txt + "\n" + '-' * len(dev_desc_txt), print_=True)

                if  valid_score < best_score:
                    
                    model.cpu()
                    torch.save({"model_state_dict": model.state_dict()},
                               os.path.join(self.ckp_save_path, f"epoch_{(i+1)}_{valid_score:5.3f}.pt"))
                    model.to(self.device)
                    best_score = valid_score
            self.scheduler.step()
            
    def train_process(self, *args, **kwargs):
        return NotImplementedError()

    def eval_model(self, *args, **kwargs):
        return NotImplementedError()
