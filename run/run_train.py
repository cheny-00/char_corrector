import os
import time

import torch
import gensim
import torch.nn as nn
import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from load_args import load_args
from model.char_embedding import CNNCharEmb
from utils.construct_dataset import LMDataset
from utils.utils import create_exp_dir, randn_sampler
from trainer.lm_trainer import LMTrainer
from model import model_table
from utils import vocab
from utils import data_collate

#############################################################################################
##  setting
#############################################################################################
args = load_args()

start_time = time.strftime('%Y%m%d-%H%M%S')
st_date, st_time = start_time.split("-")
work_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
log_dir = os.path.join(work_dir, 'logs', args.proj_name, args.model_name, st_date, st_time)
# working_files  = ["run_train.py", "load_args.py", f"model/{args.model_name}_model.py", "utils/trainer.py", "model/depressed_model.py"]
# working_files = list(map(lambda x: os.path.join(work_dir, x), working_files))
working_files = None
logging = create_exp_dir(log_dir,
                         scripts_to_save=working_files,
                         debug=args.debug)

device = torch.device(f'cuda:{args.rank}' if args.cuda else 'cpu')
if torch.cuda.is_available() and not args.cuda:
    print("detected gpu is available but not used")


save_path = os.path.join(work_dir, "save", args.model_name, st_date, st_time)
if not os.path.exists(save_path) and not args.debug:
    os.makedirs(save_path)
#############################################################################################
##  load dataset
#############################################################################################
val_split = [0.15, 0.1]

corpus_data = data_collate(args.corpus_path, args.dataset_name)

corpus_vocab = vocab.Vocab(args.vocab_path)

corpus_dataset = LMDataset(
    corpus=corpus_data,
    vocab=corpus_vocab,
    seq_len=args.seq_len,
    corpus_lines=None,
    max_len=args.max_len,
    
)

#if not os.path.exists(args.load_dataset_path):
#    torch.save(dataset, args.data)
dev_sampler, eval_sampler, train_sampler = randn_sampler(
    val_split,
    len(corpus_dataset),
    shuffle_dataset=True,
    random_seed=args.seed,
)


train_iter = DataLoader(
    corpus_dataset,
    batch_size=args.batch_size,
    drop_last=False,
    sampler=train_sampler,
)
dev_iter = DataLoader(
    corpus_dataset,
    batch_size=args.eval_batch_size,
    sampler=dev_sampler,
)
eval_iter = DataLoader(
    corpus_dataset,
    batch_size=args.eval_batch_size,
    sampler=eval_sampler,
)


model = model_table(args.model_name)
model_params = { 
    "enc_size": args.enc_size,
    "hid_size": args.hid_size,
    "dec_size": args.dec_size,
    "voc_size": len(corpus_vocab),
    
    "drop_rate": args.dropout,
    "char_kmax": args.char_kmax,
    "char_kmin": args.char_kmin,
    
    "voc_embd": None,
    "n_layer": args.n_layer,
    "device": device,
    "emb_size": 64,
    
    }

char_embd = CNNCharEmb(model_params)
if os.path.exists(args.load_char_emb_path):
    char_embd_encoder = gensim.models.KeyedVectors.load_word2vec_format(args.load_char_emb_path)
    char_embd_encoder_weight = char_embd_encoder.wv
    char_embd.encoder.weight = char_embd_encoder_weight
    
model_params['voc_embd'] = char_embd

model = model(**model_params)
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=60, eta_min= args.lr / 100)

#############################################################################################
##  loading model
############################################################################################
if args.restart:
    checkpoints = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    optim.load_state_dict(checkpoints['optim_state_dict'])
    best_score = checkpoints['score']

if args.fp8:
    model = torch.quantization.quantize_dynamic(model,
                                                {torch.nn.LSTM,
                                                 torch.nn.Linear},
                                                dtype=torch.quint8)


if args.qat:
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_fp32_fused = torch.quantization.fuse_modules(model,
                                                       [['encoder', 'decoder', 'full_connected']])

    model = torch.quantization.prepare_qat(model_fp32_fused)



#############################################################################################
##  train model
#############################################################################################
ModelTrainer = LMTrainer
tb_writter = SummaryWriter(log_dir=log_dir)
trainer_params = {
    "train_data": train_iter,
    "dev_data": dev_iter,
    "logger": logging,
    "batch_size": args.batch_size,
    "log_interval": args.log_interval,
    "eval_interval": args.eval_interval,
    "optim": optim,
    "ckp_save_path": save_path,
    "scheduler": scheduler,
    "seq_len": args.seq_len,
    "tb_writter": tb_writter,
    
}

model_trainer = ModelTrainer(**trainer_params)
criterion = nn.CrossEntropyLoss(ignore_index=corpus_vocab.pad_index)
run_process = partial(model_trainer.train_process, model=model, criterion=criterion)
model_trainer.train(model, args.epochs)

#############################################################################################
##  evaluate model
#############################################################################################

eval_start_time = time()
eval_iter = tqdm(eval_iter, desc="evaluate model", total=len(eval_iter))
eval_loss = model_trainer.eval_step(model, eval_iter)
eval_log = f"| Final Eval | Loss: {eval_loss:5.2f} | ppl: {torch.exp(eval_loss):8.2f}"
print(eval_log)


#############################################################################################
##  save model
#############################################################################################

if args.qat:
    model = torch.quantization.convert(model)

save_params = {
    "model_state_dict": model.state_dict(),
    "optim_state_dict": optim.state_dict(),
    "score": eval_loss
}
torch.save(save_params,
           os.path.join(save_path, f"{args.proj_name}.pt"))


