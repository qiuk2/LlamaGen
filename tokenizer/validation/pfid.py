import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tokenizer.tokenizer_image.vq_model import VQ_models
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse
import random
import wandb
from evaluator import Evaluator
import tensorflow.compat.v1 as tf
from utils.distributed import init_distributed_mode



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
     # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"

    # Setup env
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)


    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.dataset == 'imagenet':
        dataset = ImageFolder(args.data_path, transform=transform)
        num_fid_samples = 50000
    elif args.dataset == 'coco':
        dataset = SingleFolderDataset(args.data_path, transform=transform)
        num_fid_samples = 5000
    else:
        raise Exception("please check dataset")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )    

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    global_batch_size = args.global_batch_size

    if dist.get_rank() == 0:
        # Perform t-SNE dimensionality reduction
        wandb_tracker = wandb.init(project='PFID', name=f'LlamaGen')

    # create and load model
    vae = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    )
    #
    # # load checkpoints
    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vae.load_state_dict(checkpoint["model"])
        if args.ema:
            vae.load_state_dict(checkpoint["ema"])

    for p in vae.parameters(): p.requires_grad_(False)
    vae = vae.to(device)
    print(f'prepare finished.')

    total = 0
    gt, samples = [], []
    loader = tqdm(loader)
    for step, (x, label) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True, dtype=torch.long)
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16,
                                cache_enabled=True):  # using bfloat16 can be faster
                # sample = vae.img_to_reconstructed_img(x, noise=args.top_p, level=args.top_k, resid=True)
                sample = vae.img_to_reconstructed_img(x, level=args.top_k, noise=args.top_p)
            
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
            x = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
                        
            sample = torch.cat(dist.nn.all_gather(sample), dim=0)
            x = torch.cat(dist.nn.all_gather(x), dim=0)
            samples.append(sample.to("cpu", dtype=torch.uint8).numpy())
            gt.append(x.to("cpu", dtype=torch.uint8).numpy())
            if dist.get_rank() == 0 and step % 10 == 0:
                sample = sample.permute(0, 3, 1, 2)
                x = x.permute(0, 3, 1, 2)
                show_images = torch.cat((x[:4], sample[:4]), dim=0)
                wandb_tracker.log({"recon_images": [wandb.Image(show_images)]}, step=step)
    
    dist.barrier()
    if dist.get_rank() == 0:
        samples = np.concatenate(samples, axis=0)
        gt = np.concatenate(gt, axis=0)
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True

        evaluator = Evaluator(tf.Session(config=config))
        evaluator.warmup()
        ref_acts = evaluator.read_activations(gt)
        ref_stats, _ = evaluator.read_statistics(gt, ref_acts)
        sample_acts = evaluator.read_activations(samples)
        sample_stats, _ = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)
        with open("result.txt", "a") as f:
            f.write(f"top_k: {args.top_k}, top_p: {1 - args.top_p}, FID: {FID:.4f}\n")
        print("Results saved to result.txt")

        print(f"Select from top {args.top_k} with prob {args.top_p} FID: {FID}")
    
    dist.barrier()
    print("Done")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco'], default='imagenet')
    parser.add_argument("--global-batch-size", type=int, default=768)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.0)

    args = parser.parse_args()
    main(args)