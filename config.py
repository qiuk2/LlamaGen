project = 'ilo-noquota-p4de'
job_name = 'msvq-var-train-99'
num_nodes = 4
num_gpus = 8
job_file = 's3://xangl9867'
WANDB_API_KEY = 'eed03e9548474fc9bccb341783e5704c46647181'


with open('private_key', 'r') as file:
    private_key_content = file.read()
