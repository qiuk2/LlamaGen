from config import *
from common import run_command, setup_aws_credentials, read_aws_credentials
import ray
import argparse

parser = argparse.ArgumentParser(description="Download and extract a specific folder from Hugging Face dataset repository.")
parser.add_argument("--target_folder", type=str, required=True, choices=["VQGAN", '1d-tokenizer', 'VQGAN-LC', 'IBQ'], help="The specific subfolder to download and extract.")
args = parser.parse_args()

ray.init()

## get master IP
master_ip = run_command('hostname -I', 0, False)
print(master_ip)
master_ip = master_ip.replace("\n", "")

## training command and launch
#TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO
cmd = ('nohup torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=master_ip --master_port=12345 '
       'autoregressive/train/train_c2i.py '
       f'--cloud-save-path {args.target_folder} --code-path /mnt/localssd/imagenet_code_c2i_flip_ten_crop --image-size 256 --gpt-model GPT-L --no-local-save --ckpt-every 20000 ')

# 'torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 inference.py --data_path /mnt/localssd/ImageNet2012/ --workers 12 --encoder_model vit_base_patch14_dinov2.lvd142m  --decoder_model vit_base_patch14_dinov2.lvd142m --product_quant 2 --semantic_guide dinov2 --num_latent_tokens 121 --codebook_embed_dim 32 --codebook_size 4096  --v_patch_nums 1 1 2 3 3 4 5 6 8 11 --pn 1_1_2_3_3_4_5_6_8_11 --patch_size 11 --sem_half True --cfg 3.0'
training_command = f"""conda activate var; cd LlamaGen; {cmd} > train_output.out 2>&1 &"""
training_command = training_command.replace('master_ip', master_ip)
training_command = training_command.replace('nnodes=1', f'nnodes={num_nodes}')
training_command = training_command.replace('nproc_per_node=8', f'nproc_per_node={num_gpus}')

export = 'export FI_PROVIDER=efa; export FI_EFA_USE_DEVICE_RDMA=1; export NCCL_PROTO=simple'

@ray.remote
def train(i):
    ## kill Python processes
    run_command('pkill python -9', i)
    
    # ## set up AWS credentials
    # aws_creds = setup_aws_credentials(i, '~/.aws/aws_credentials_data')
    # # run_command('pip install timm', i)

    # ## data-related preparation
    # run_command('rm MFM_mim/aws_credentials_1859', i)
    # aws_cred_path_test = 'MFM_mim/aws_credentials_1859'
    # for c in aws_creds:
    #     run_command(f'echo {c} >> {aws_cred_path_test}', i)
    # run_command('cd MFM_mim; python mfm_projects/unified_masking_datasets/mfm/dataset_mfm_v3_3.py', i, timeout=30) #dataset_mfm_v3_3.py, dataset_mfm_v2_10.py
    # export_with_aws = export
    # for c in aws_creds:
    #     c = c.replace('aws_access_key_id', 'AWS_ACCESS_KEY_ID').replace('aws_secret_access_key', 'AWS_SECRET_ACCESS_KEY')
    #     export_with_aws += '; export ' + c

    ## pretrained backbone
    # run_command('rm MFM_mim/vit-huge-p16_sam-pre_3rdparty_sa1b-1024px_20230411-3f13c653.pth', i)
    # run_command("cd MFM_mim; aria2c -x16 -s16 https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-huge-p16_sam-pre_3rdparty_sa1b-1024px_20230411-3f13c653.pth", i)
    #run_command("cd MFM_mim; aria2c -x16 -s16 https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt", i)

    # run_command('aws s3 cp s3://mfm-checkpoints/mfm_v0.11.5/iter_15000.pth MFM_mim/', i)
    node_training_command = training_command.replace('--node_rank=0', f'--node_rank={i}')
    node_training_command = f'export OMP_NUM_THREADS=1; {node_training_command}'
    run_command(node_training_command, i)
# LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64

futures = [train.remote(i) for i in range(num_nodes)]
print(ray.get(futures))
