#!/bin/bash
mkdir log

python train_main.py --use_gca --dataset demo1,demo2 --data_dir ./data/ --use_bert --bert_model=./bert-base-chinese --word_embeddings=data/embedding.txt --max_seq_len=500 --batch_size=16 --num_epoch=2 --warmup_proportion=0.1 --learning_rate=1e-5
