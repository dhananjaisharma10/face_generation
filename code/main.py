import os
import torch
import config
import argparse
import numpy as np
import matplotlib.pyplot as plt
from runner import Runner
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Face Generation.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='\'train\' or \'test\' mode.')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID to load model.')
    parser.add_argument('--g_model_name', type=str, default=None, help='Name of model file for Generator.')
    parser.add_argument('--d_model_name', type=str, default=None, help='Name of model file for Discrimintor.')
    return parser.parse_args()

def create_dir(run_id):
    # Create directories if not exist.
    path = os.path.join(config.model_save_dir, '{}/Generator'.format(run_id))
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(config.model_save_dir, '{}/Discriminator'.format(run_id))
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(config.result_dir, '{}'.format(run_id))
    if not os.path.exists(path):
        os.makedirs(path)

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
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(config.result_dir,'{}/losses.jpeg'.format(run_id)), dpi=400, bbox_inches='tight')
    plt.close()

def plot_images(epoch, img, run_id):
    fig = plt.figure(figsize=(10,5))
    plt.axis("off")
    plt.title("Fake Image {}".format(epoch))
    plt.imshow(np.transpose(img_list,(1,2,0))) # plot the latest epoch
    plt.savefig(os.path.join(config.result_dir,'{}/images_{}.jpeg'.format(run_id, epoch)), dpi=400, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True   # boost speed.
    args = parse_args()     # Parse args.
    setup_random_seed()     # Set random seed for shuffle.

    print('='*20)
    if args.mode == 'train':
        runner = Runner()

        # Prepare directories.
        dt = datetime.now()
        run_id = dt.strftime('%m_%d_%H_%M')
        create_dir(run_id)      # Create relevant directories.

        n_epochs = config.n_epochs
        G_losses = []
        D_losses = []

        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch+1,n_epochs))
            d_loss, g_loss, result = runner.train_model()

            G_losses.append(g_loss)
            D_losses.append(d_loss)
            plot_loss(G_losses, D_losses, run_id) # loss image
            plot_images(epoch+1, result, run_id) # images

            # Checkpoint the model after each epoch.
            d_loss, g_loss= '%.3f'%(d_loss), '%.3f'%(g_loss)
            model_path_G = os.path.join('{}/{}/Generator'.format(config.model_save_dir, run_id), \
                         'G_model_{}_d_{}_g_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), d_loss, g_loss))
            model_path_D = os.path.join('{}/{}/Discriminator'.format(config.model_save_dir, run_id), \
                         'D_model_{}_d_{}_g_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), d_loss, g_loss))
            torch.save(runner.G.state_dict(), model_path_G)
            torch.save(runner.D.state_dict(), model_path_D)
            print('='*20)

    elif args.mode == 'test':
        # For loading pre-trained model.
        g_path = os.path.join('{}/{}/Generator'.format(config.model_save_dir, args.run_id), args.g_model_name)
        #d_path = os.path.join('{}/{}/Discriminator'.format(config.model_save_dir, args.run_id), args.d_model_name)
        runner = Runner(reload_model=True, g_model_path=g_path)
        result = runner.test_model()
        plot_images(9999, [result], args.run_id)
        print('='*20)
