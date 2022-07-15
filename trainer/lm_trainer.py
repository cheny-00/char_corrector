import torch
from tqdm import tqdm

from trainer.trainer import BaseTrainer

class LMTrainer(BaseTrainer):
    
    def __init__(self, optim, seq_len, batch_size, emb_dim, train_data, dev_data, tb_writer, logger, log_interval, eval_interval, scheduler=None, ckp_save_path=None, train_step=0, device='cpu'):
        super().__init__(optim, seq_len, batch_size, emb_dim, train_data, dev_data, tb_writer, logger, log_interval, eval_interval, scheduler, ckp_save_path, train_step, device)
    
    def train_process(self, model, data, **kwargs):
        criterion = kwargs['criterion']
        hidden = model.init_hidden(self.batch_size, self.seq_len)
        hidden = to_device(hidden, self.device)
        total_loss = 0
        n_data = 0
        for x, y in self.get_batch(data):
            if x.shape[1] != self.seq_len:
                break
            
            self.optim.zero_grad()
            hidden = repackage_hidden(hidden)
            
            logit, hidden = model(x, hidden)
            loss = criterion(logit, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            
            total_loss += loss.float().item()
            n_data += 1         
            """
            # SGD
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
            """
            
            self.optim.step()
        mean_loss = total_loss / n_data
        return mean_loss
    
    def eval_model(self, model, eval_iter=None, **kwargs):
        
        model.eval()
        criterion = kwargs['criterion']
        ret = dict()
        if eval_iter is None:
            eval_iter = self.dev_data
        
        dev_total_loss = 0
        
        with torch.no_grad():
            for data in tqdm(eval_iter, desc="do eval step"):
                hidden = model.init_hidden(data['x'].shape[0], self.seq_len)
                hidden = to_device(hidden, self.device)
                
                n_data = 0
                batch_loss = 0 
                for x, y in self.get_batch(data):
                    if x.shape[1] != self.seq_len:
                        break
                    logit, hidden = model(x, hidden)
                    loss = criterion(logit, y)
                    hidden = repackage_hidden(hidden)
                    
                    batch_loss += loss.item()
                    n_data += 1
                dev_total_loss += batch_loss / n_data
        ret['valid_score'] = dev_total_loss / len(self.dev_data)
        return ret
    
    def get_batch(self, data):
        """
        Subdivides the source into chunks of length bptt.
        The chunks are along dimension 0 (length of each row is n_seq).
        """
        seq_len = self.seq_len
        for i in range(0, len(data['x']), seq_len-1):
            # seq_len = min(self.seq_len, len(source)-1-i)
            x = data['x'][:, i:i+seq_len].to(self.device)
            y = data['x'][:, i+1:i+1+seq_len].reshape(-1).to(self.device)
            yield x, y


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def to_device(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    else:
        return tuple(to_device(_v, device) for _v in v)