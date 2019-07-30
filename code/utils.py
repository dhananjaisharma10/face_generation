import os
import sys
import torch
import config
import numpy as np
import matplotlib.pyplot as plt

def create_dir(run_id):
    # Create directories if not exist.
    path = os.path.join(config.models_dir, '{}'.format(run_id))
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(config.result_dir, '{}'.format(run_id))
    if not os.path.exists(path):
        os.makedirs(path)

def dump_run_config(run_id):
    run_config_file = os.path.join(config.models_dir, run_id, '{}.config'.format(run_id))
    with open(run_config_file, 'w') as f:
        f.write('Sys Args:\n\n')
        f.write(str(sys.argv))
        f.write('\n\n')
        with open('config.py','r') as cf:
            lines = cf.readlines()
        for line in lines:
            f.write(line)

def setup_random_seed():
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

def plot_loss(G_losses, D_losses, run_id):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(config.result_dir,'{}/losses.jpeg'.format(run_id)), dpi=400, bbox_inches='tight')
    plt.close()

def plot_images(title, img, run_id, seq, mode='train'):
    fig = plt.figure(figsize=(10,5))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(img,(1,2,0)))
    result_path = os.path.join(config.result_dir, run_id, mode, 'images_{}.jpeg'.format(seq))
    result_dir = os.path.dirname(result_path)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    plt.savefig(result_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
