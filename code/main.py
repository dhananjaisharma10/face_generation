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

def plot_images(epoch, img_list, run_id):
    # Grab a batch of real images from the dataloader
    ##
    # Plot the fake images from the last epoch
    fig = plt.figure(figsize=(10,5))
    plt.axis("off")
    plt.title("Fake Image {}".format(len(img_list)))
    plt.imshow(np.transpose(img_list[-1],(1,2,0))) # plot the latest epoch
    plt.savefig(os.path.join(config.result_dir,'{}/images_{}.jpeg'.format(run_id, epoch)), dpi=400, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True   # boost speed.
    dt = datetime.now()
    run_id = dt.strftime('%b-%d_%H:%M')
    #if not os.path.exists('./experiments'):
    #    os.mkdir('./experiments')
    #os.mkdir('./experiments/%s' % run_id)
    # print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)
    args = parse_args()     # Parse args.
    create_dir(run_id)            # Create relevant directories.
    setup_random_seed()     # Set random seed for shuffle.
    runner = Runner()
    n_epochs = config.n_epochs
    G_losses = []
    D_losses = []
    img_list = []
    if args.mode == 'train':
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch+1,n_epochs))
            d_loss, g_loss, result = runner.train_model()

            G_losses.append(g_loss)
            D_losses.append(d_loss)
            plot_loss(G_losses, D_losses, run_id) # loss image
            img_list.append(result)
            plot_images(epoch+1, img_list, run_id) # images
            # Checkpoint the model after each epoch.
            d_loss, g_loss= '%.3f'%(d_loss), '%.3f'%(g_loss)
            model_path_G = os.path.join('{}/{}/Generator'.format(config.model_save_dir, run_id), \
                         'G_model_{}_d_{}_g_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), d_loss, g_loss))
            model_path_D = os.path.join('{}/{}/Discriminator'.format(config.model_save_dir, run_id), \
                         'D_model_{}_d_{}_g_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), d_loss, g_loss))
            torch.save(runner.G.state_dict(), model_path_G)
            torch.save(runner.D.state_dict(), model_path_D)
            print('='*20)

        # runner.test_model()
        # print('='*20)
