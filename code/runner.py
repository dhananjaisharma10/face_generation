import os
import time
import torch
import config
import torch.optim as optim
from dataset import get_loader
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from model import Discriminator, Generator

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Runner(object):
    def __init__(self):
        super(Runner, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.D = Discriminator(ngpu=config.ngpu)#, nc=config.d_in_channels, ndf=config.d_conv_dim)
        self.G = Generator(ngpu=config.ngpu)#, nz=config.c_dim,
                            # ngf=config.g_conv_dim, nc=config.g_out_channels)
        self.D.apply(weights_init)
        self.G.apply(weights_init)
        self.D, self.G = self.D.to(self.device), self.G.to(self.device)
        self.train_loader = get_loader(config.image_dir, config.attr_path,
                                        crop_size=config.crop_size, image_size=config.image_size,
                                        batch_size=config.train_batch_size, mode='train',
                                        num_workers=config.num_workers)
        self.test_loader = get_loader(config.image_dir, config.attr_path,
                                        crop_size=config.crop_size, image_size=config.image_size,
                                        batch_size=config.test_batch_size, mode='test',
                                        num_workers=config.num_workers)
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=config.g_lr, weight_decay=config.g_wd)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=config.d_lr, weight_decay=config.d_wd)
        self.lambda_cls = config.lambda_cls
        self.lambda_gp = config.lambda_gp
        self.n_critic = config.n_critic
        self.result_dir = config.result_dir
        self.nz = config.c_dim
        self.criterion = nn.BCELoss()
        self.fixed_feats = None
        for _, (_, feats) in enumerate(self.test_loader, 0):
            self.fixed_feats = feats.view(feats.size(0), feats.size(1), 1, 1).to(self.device)
            break

    def classification_loss(self, preds, targets):
        return F.binary_cross_entropy_with_logits(preds, targets)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def train_model(self):
        self.D.train()
        self.G.train()
        # Training Loop
        # Lists to keep track of progress
        img_list = []
        # G_losses = []
        # D_losses = []

        # For each epoch
        d_running_loss = 0
        
        d_running_p_x = 0
        d_running_p_gz1 = 0
        d_running_p_gz2 = 0
        
        g_running_loss = 0
        iters = 0
        # For each batch in the dataloader
        for _, (targets, feats) in enumerate(self.train_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            self.D.zero_grad()
            # Format batch
            feats, targets = feats.to(self.device), targets.to(self.device)
            real_cpu = feats
            b_size = real_cpu.size(0)
            feats = feats.view(b_size, self.nz, 1, 1)
            label = torch.full((b_size,), config.real_label, device=self.device)
    #         label = feats
            # Forward pass real batch through D

            output = self.D(targets)
            output = output.view(-1)
            # Calculate loss on all-real batch

            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            d_running_p_x += D_x
            ## Train with all-fake batch
            # Generate batch of latent vectors
    #         noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
    #         fake = netG(noise)
            fake = self.G(feats)
            label.fill_(config.fake_label)
            # Classify all fake batch with D
            output = self.D(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            d_running_p_gz1 += D_G_z1
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            d_running_loss += errD.item()
            # Update D
            self.d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.G.zero_grad()
            label.fill_(config.real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.D(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            g_running_loss += errG.item()
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            d_running_p_gz2 += D_G_z2
            
            # Update G
            self.g_optimizer.step()
            
            # Output training stats
            # if i % 50 == 0:
            #     print('[%d/%d][%d/%d] Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #         % (epoch, num_epochs, i, len(self.train_loader),
            #             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            
            # Save Losses for plotting later
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise

            iters += 1

            if (iters % 500 == 0):
                with torch.no_grad():
                    fake = self.G(self.fixed_feats).detach().cpu()
                    img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))
        
        j = len(self.train_loader)

        print('Running -> Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % d_running_loss / j, g_running_loss/ j, d_running_p_x/ j,
                        d_running_p_gz1/ j, d_running_p_gz2/ j)
            
        d_running_loss /= j
        g_running_loss /= j

        return d_running_loss, g_running_loss, img_list

    # def test_model(self):
    #     with torch.no_grad():
    #         self.D.eval()
    #         self.G.eval()
    #         start_time = time.time()
    #         for batch_idx, (imgs, targets) in enumerate(self.test_loader):
    #             imgs, targets = imgs.to(self.device), targets.to(self.device)
    #             imgs_fake = self.G(targets)
    #             imgs_concat = torch.cat([imgs, imgs_fake], dim=0)
    #             result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(batch_idx+1))
    #             save_image(self.denorm(imgs_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             print('Test Iteration: %d/%d, Saved: %s' % (batch_idx+1, len(self.test_loader), result_path), end="\r", flush=True)
    #         end_time = time.time()
    #         print('\nTest Completed. Time: %d s' % (end_time - start_time))


        
