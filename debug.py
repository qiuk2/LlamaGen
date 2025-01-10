from huggingface_hub import HfApi, upload_file

repo_name = "qiuk6/PFID" 
hf_token = "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"

cloud_checkpoint_path = "/home/biometrics/kaiqiu/VQGAN-LC/vqgan-gpt-lc/VQGAN-LC/2025-01-05-00-02-36/000-GPT-B/checkpoints/1500000.pt"
upload_file(
    path_or_fileobj=cloud_checkpoint_path, 
    path_in_repo="/".join(cloud_checkpoint_path.split("/")[-5:]),     # Target path in the repository
    repo_id=repo_name,          
    token=hf_token,
)