
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
