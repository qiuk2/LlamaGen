#!/bin/bash

top_p_values=(0.1 0.2 0.3 0.4 0.5) # 根据需要调整
top_k_values=(200 280 360)       # 根据需要调整

# 定义固定的其他参数
vq_ckpt="vq_ds16_c2i.pt"
data_path="/home/biometrics/kaiqiu/ImageNet2012/val/"
globat_batch_size=128

# 开始网格搜索
for top_p in "${top_p_values[@]}"; do
  for top_k in "${top_k_values[@]}"; do
    echo "Running with top_p=$top_p and top_k=$top_k"
    
    # 构建运行命令
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 tokenizer/validation/pfid.py \
      --vq-ckpt $vq_ckpt \
      --data-path $data_path \
      --global-batch-size $globat_batch_size \
      --top_k $top_k \
      --top_p $top_p
  done
done
