import os
import torch
import config
import argparse
import numpy as np
from runner import Runner
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Face Generation.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='\'train\' or \'test\' mode.')
    return parser.parse_args()

def create_dir():
    # Create directories if not exist.
    if not os.path.exists(config.model_save_G_dir):
        os.makedirs(config.model_save_G_dir)
    if not os.path.exists(config.model_save_D_dir):
        os.makedirs(config.model_save_D_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

def setup_random_seed():
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

if __name__ == "__main__":
    args = parse_args()     # Parse args.
    create_dir()            # Create relevant directories.
    setup_random_seed()     # Set random seed for shuffle.
    runner = Runner()
    n_epochs = config.n_epochs
    if args.mode == 'train':
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch+1,n_epochs))
            d_loss, g_loss = runner.train_model()
            # Checkpoint the model after each epoch.
            d_loss, g_loss= '%.3f'%(d_loss), '%.3f'%(g_loss)
            model_path_G = os.path.join(config.model_save_G_dir, \
                        'G_model_{}_d_{}_g_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), d_loss, g_loss))
            model_path_D = os.path.join(config.model_save_D_dir, \
                        'D_model_{}_d_{}_g_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), d_loss, g_loss))
            torch.save(runner.G.state_dict(), model_path_G)
            torch.save(runner.D.state_dict(), model_path_D)
            print('='*20)

        runner.test_model()
        print('='*20)
