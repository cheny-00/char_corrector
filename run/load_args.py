
import argparse

def load_args():
    parser = argparse.ArgumentParser(description="train model")

    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--epochs", type=int, default=512, help="number of epochs for training")
    parser.add_argument("--seed", type=int, default=44, help="random seed")
    parser.add_argument("--dropout", type=float, default=0.25, help="setting dropout probability")
    parser.add_argument("--debug", action="store_true", help="open debug mode")
    parser.add_argument("--restart", action="store_true", help="restart model")

    parser.add_argument("--ckpt_path", type=str, default="../checkpoints/", help="path to load checkpoints")
    parser.add_argument("--cuda", action="store_false", help="use gpu")
    parser.add_argument("--log_interval", type=int, default=200, help="report interval")
    parser.add_argument("--eval_interval", type=int, default=8, help="the number of epochs to evaluation interval")

    parser.add_argument("--batch_size", type=int, default=512, help="batch size for train")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="batch size for evaluation")
    parser.add_argument("--corpus_path", type=str, default="/home/cy/workspace/datasets/new_240hz_data", help="loading corpus path")
    parser.add_argument("--vocab_path", type=str, default="../data/", help="vocab_path")
    

    parser.add_argument("--dataset_name", type=str, default="oscar", help="dataset name")
    parser.add_argument("--proj_name", type=str, default="rnnlm", help="project name")
    parser.add_argument("--model_name", type=str, default="rnnlm", help="select model")

    parser.add_argument("--rank", type=str, default="0", help="gpu rank")
    parser.add_argument("--qat", action="store_true", help="qat")
    parser.add_argument("--fp8", action="store_true", help="float 8bit")
    parser.add_argument("--n_data", type=int, default=-1, help="numbers of using data, -1 for using all data")
    
    parser.add_argument("--seq_len", type=int, default=64, help="sequence length")
    parser.add_argument("--enc_size", type=int, default=128, help="hidden size of encoder")
    parser.add_argument("--dec_size", type=int, default=64, help="hidden size of decoder")
    parser.add_argument("--hid_size", type=int, default=64, help="hidden size")
    parser.add_argument("--voc_size", type=int, default=64, help="hidden size of vocab embedding")
    parser.add_argument("--n_layer", type=int, default=1, help="numbers of unit layers")
    
    
    args = parser.parse_args()

    return args