from data import get_loader

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['CelebA']:
            solver.test()


# Model configuration.
'c_dim: dimension of domain labels (1st dataset)'
c_dim = 5

'celeba_crop_size: crop size for the CelebA dataset'
celeba_crop_size = 178

'image_size: image resolution'
image_size = 128

'g_conv_dim: number of conv filters in the first layer of G'
g_conv_dim = 64

'd_conv_dim: number of conv filters in the first layer of D'
d_conv_dim = 64

'g_num_blocks: number of residual blocks in G'
g_num_blocks = 6

'd_repeat_num: number of strided conv layers in D'
d_repeat_num = 6

'lambda_cls: weight for domain classification loss'
lambda_cls = 1

'lambda_rec: weight for reconstruction loss'
lambda_rec = 10

'lambda_gp: weight for gradient penalty'
lambda_gp = 10


# Training configuration.
'dataset: CelebA'
dataset = 'CelebA'

'train_batch_size: mini-batch size for training'
train_batch_size = 16

'test_batch_size: mini-batch size for testing'
test_batch_size = 16

'num_iters: number of total iterations for training D'
num_iters = 200000

'num_iters_decay: number of iterations for decaying lr'
num_iters_decay = 100000

'g_lr: learning rate for G'
g_lr = 0.0001

'd_lr: learning rate for D'
d_lr = 0.0001

'd_channels: list of in/out channels for conv layers in D'
d_channels = [3, 64, 128, 256, 512, 1]

'd_slope: slope of leaky relu'
d_slope = 0.2


'n_critic: number of D updates per each G update'
n_critic = 5

'beta1: beta1 for Adam optimizer'
beta1 = 0.5

'beta2: beta2 for Adam optimizer'
beta2 = 0.999

'resume_iters: resume training from this step'
resume_iters = None

'selected_attrs: selected attributes for the CelebA dataset'
selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']


# Miscellaneous.
'num_workers'
num_workers = 1

'mode: train/test'
mode = 'train'

'use_tensorboard'
use_tensorboard = True

# Directories.
celeba_image_dir = 'data/celeba/images'

attr_path = 'data/celeba/list_attr_celeba.txt'

log_dir = '../logs'

model_save_dir = '../models'

sample_dir = '../samples'

result_dir = '../results'


# Step size.
log_step = 10

sample_step = 1000

model_save_step = 10000

lr_update_step = 1000

n_epochs = 100
