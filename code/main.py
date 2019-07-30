import os
import sys
import time
import torch
import config
import argparse
from utils import *
from runner import Runner
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Face Generation.')
    parser.add_argument('--run_id', type=str, required="--model_name" in sys.argv, help='Run ID to load model.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='\'train\' or \'test\' mode.')
    parser.add_argument('--model_name', type=str, default=None, help='Name of model file to be reloaded.')
    return parser.parse_args()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True   # boost speed.
    args = parse_args()     # Parse args.
    setup_random_seed()     # Set random seed for shuffle.

    print('='*20)
    if args.mode == 'train':
        runner = Runner(args=args)
        # Prepare directories.
        dt = datetime.now()
        run_id = dt.strftime('%m_%d_%H_%M')
        create_dir(run_id)
        # dump args to file.
        dump_run_config(run_id)
        # Start training.
        n_epochs = config.n_epochs
        G_losses = []
        D_losses = []
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch+1,n_epochs))
            d_loss, g_loss, result = runner.train_model()
            # Plot losses.
            G_losses.append(g_loss)
            D_losses.append(d_loss)
            plot_loss(G_losses, D_losses, run_id)   # loss image
            plot_images("Fake Image : Epoch {}".format(epoch+1), result, run_id, epoch+1)    # images
            # Checkpoint the model after each epoch.
            d_loss, g_loss= '%.3f'%(d_loss), '%.3f'%(g_loss)
            model_path = os.path.join('{}/{}'.format(config.models_dir, run_id), \
                         'model_{}_d_{}_g_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), d_loss, g_loss))
            model_dict = {'Generator':runner.G.state_dict(),
                            'Discriminator':runner.D.state_dict()
                            }
            torch.save(model_dict, model_path)
            print('='*20)

    elif args.mode == 'test':
        runner = Runner(args=args)
        result = runner.test_model()
        print('='*20)
