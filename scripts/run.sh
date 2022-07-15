ln="en"
python ../run/run_train.py  \
--lr 0.0055  \
--epochs    64 \
--seed  614    \
--dropout 0.25  \
--ckpt_path /mnt/data/tmp_workspace/char_corrector/ckpt \
--batch_size 256    \
--max_len   512 \
--seq_len   128  \
--rank  6  \
--corpus_path   /mnt/data/corpus_dataset/indices_data/en_indices_all.arrow \
--vocab_path    ../data/${ln}_dictionary.txt    \
--load_char_emb_path ../data/en_char_emb.wv 
# --debug
# --cuda  
# --debug 