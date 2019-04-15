# Model configuration
# c_dim: dimension of domain labels (1st dataset)
c_dim = 40

# crop_size: crop size for the CelebA dataset
crop_size = 178

# image_size: image resolution
image_size = 128

# g_conv_dim: number of conv filters in the first layer of G
g_conv_dim = 64

# d_conv_dim: number of conv filters in the first layer of D
d_conv_dim = 64

# g_num_blocks: number of residual blocks in G
g_num_blocks = 6

# g_repeat_num: number of conv layers for up/down-sampling in G
g_repeat_num = 2

# d_repeat_num: number of strided conv layers in D
d_repeat_num = 6

# lambda_cls: weight for domain classification loss
lambda_cls = 1

# lambda_rec: weight for reconstruction loss
lambda_rec = 10

# lambda_gp: weight for gradient penalty
lambda_gp = 10


# Training configuration
# train_batch_size: mini-batch size for training
train_batch_size = 16

# test_batch_size: mini-batch size for testing
test_batch_size = 1

# num_workers
num_workers = 1

# n_epochs: number of training epochs
n_epochs = 20

# g_lr: learning rate for G
g_lr = 0.0001

# d_lr: learning rate for D
d_lr = 0.0001

# g_wd: weight decay for G
g_wd = 0

# g_wd: weight decay for D
d_wd = 0

# d_channels: list of in/out channels for conv layers in D
d_channels = [3, 64, 128, 256, 512]

# d_slope: slope of leaky relu
d_slope = 0.2

# n_critic: number of D updates per each G update
n_critic = 5

# beta1: beta1 for Adam optimizer
beta1 = 0.5

# beta2: beta2 for Adam optimizer
beta2 = 0.999

# selected_attrs: selected attributes for the CelebA dataset
selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']


# Miscellaneous
# use_tensorboard
use_tensorboard = True

# random_seed: to give consistent results
random_seed = 1111

# Directories
image_dir = './../data/images'

attr_path = './../data/list_attr_celeba.txt'

log_dir = './../logs'

model_save_G_dir = './../models/Generator'
model_save_D_dir = './../models/Discriminator'


sample_dir = './../samples'

result_dir = './../results'


# Step size
log_step = 10

sample_step = 1000

model_save_step = 10000

lr_update_step = 1000
