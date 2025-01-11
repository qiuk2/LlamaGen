import os
import zipfile
from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser(description="Download and extract a specific folder from Hugging Face dataset repository.")
parser.add_argument("--target_folder", type=str, required=True, choices=["VQGAN", '1d-tokenizer', 'VQGAN-LC', 'IBQ'], help="The specific subfolder to download and extract.")
args = parser.parse_args()

# 配置
repo_id = "qiuk6/PFID-data"  # 替换为您的仓库名称
local_dir = "/mnt/localssd"  # 本地保存路径
target_folder = f"{args.target_folder}-data" # 目标子文件夹
hf_token = "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"

# 下载特定文件夹
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",  # 数据仓库类型
    local_dir=local_dir,  # 本地保存路径
    allow_patterns=[f"{target_folder}/"],  # 仅下载目标文件夹及其内容
    token=hf_token
)

local_dir = os.path.join(local_dir, target_folder)

for subfolder1 in os.listdir(local_dir):
    for subfolder2 in os.listdir(os.path.join(local_dir, subfolder1)):
        zip_file = os.path.join(local_dir, subfolder1, subfolder2)
        output_dir = zip_file[:-4]
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)


print(f"子文件夹 {target_folder} 已成功下载到 {local_dir}")
