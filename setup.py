from config import *
from common import run_command, setup_aws_credentials
import ray
import argparse

parser = argparse.ArgumentParser(description="Download and extract a specific folder from Hugging Face dataset repository.")
parser.add_argument("--target-folder", type=str, nargs='+', required=True, choices=["VQGAN", '1d-tokenizer', 'VQGAN-LC', 'IBQ'], help="The specific subfolder to download and extract.")
args = parser.parse_args()

ray.init()

@ray.remote
def mfm_setup(i):
    run_command('pkill python -9', i)

    ## directories and permissions
    # run_command('sudo mkdir /home/xangl', i, False)
    # run_command('sudo chown -R xangl /home/xangl', i, False)
    # run_command('sudo chown -R xangl /home/user', i, False)
    # run_command('sudo chown -R xangl /opt/venv/', i, False)
    
    # Escape the private key content for safe shell usage
    escaped_private_key_content = private_key_content.replace('"', '\\"')

    # Command to write the private key to remote node
    copy_command = f'echo \'{escaped_private_key_content}\' > /home/xiangl/.ssh/id_rsa && chmod 600 /home/xiangl/.ssh/id_rsa'
    run_command(copy_command, i)
    run_command('ssh -o StrictHostKeyChecking=no -T git@github.com', i)
    run_command(f'wandb login {WANDB_API_KEY}', i)

    run_command('git clone git@github.com:qiuk2/LlamaGen.git', i)
    run_command('cd LlamaGen; conda env create -f environment.yml ', i)

    # prepare data
    for target_folder in args.target_folder:
        run_command(f'conda activate var; cd LlamaGen; python prepare_data.py --target_folder {target_folder}', i)


futures = [mfm_setup.remote(i) for i in range(num_nodes)]
print(ray.get(futures))
