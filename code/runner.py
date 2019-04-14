import os
import time
import torch
import config
import torch.optim as optim
from dataset import get_loader
import torch.nn.functional as F
from torchvision.utils import save_image
from model import Discriminator, Generator

class Runner(object):
    def __init__(self):
        super(Runner, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.D = Discriminator(image_size=config.image_size, slope=config.d_slope,
                                conv_dim=config.d_conv_dim, c_dim=config.c_dim, repeat_num=config.d_repeat_num)
        self.G = Generator(image_size=config.image_size, conv_dim=config.g_conv_dim,
                            c_dim=config.c_dim, num_blocks=config.g_num_blocks, repeat_num=config.g_repeat_num)
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
        d_running_loss = 0.0
        g_running_loss = 0.0
        start_time = time.time()
        for batch_idx, (imgs, targets) in enumerate(self.train_loader):
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            # TRAIN THE DISCRIMINATOR.
            # Compute loss with real images.
            #out_src, out_cls = self.D(imgs)
            out_src = self.D(imgs)
            d_loss_real = - torch.mean(out_src)
            #d_loss_cls = self.classification_loss(out_cls, targets)
            # Compute loss with fake images.
            imgs_fake = self.G(targets)
            #out_src, _ = self.D(imgs_fake.detach())
            out_src = self.D(imgs_fake.detach())
            d_loss_fake = torch.mean(out_src)
            # Compute loss for gradient penalty. TBD: Read more about this.
            #alpha = torch.rand(imgs.size(0), 1, 1, 1).to(self.device)
            #imgs_hat = (alpha * imgs.data + (1 - alpha) * imgs_fake.data).requires_grad_(True)
            #out_src, _ = self.D(imgs_hat)
            #d_loss_gp = self.gradient_penalty(out_src, imgs_hat)
            # Backward and optimize.
            #d_loss = d_loss_real + d_loss_fake + (self.lambda_cls * d_loss_cls) + (self.lambda_gp * d_loss_gp)
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            d_running_loss += d_loss.item()
            self.d_optimizer.step()
            # TRAIN THE GENERATOR.
            if (batch_idx+1) % self.n_critic == 0:
                imgs_fake = self.G(targets)
                #out_src, out_cls = self.D(imgs_fake)
                out_src = self.D(imgs_fake)
                g_loss_fake = - torch.mean(out_src)
                #g_loss_cls = self.classification_loss(out_cls, targets)
                # Backward and optimize.
                g_loss = g_loss_fake
                self.reset_grad()
                g_loss.backward()
                g_running_loss += g_loss.item()
                self.g_optimizer.step()
            # Logging
            print('Train Iteration: %d/%d D_Loss = %5.4f G_Loss = %5.4f' % \
                (batch_idx+1, len(self.train_loader), (d_running_loss/(batch_idx+1)), \
                (g_running_loss/(batch_idx+1))), end="\r", flush=True)
        end_time = time.time()
        d_running_loss /= len(self.train_loader)
        g_running_loss /= len(self.train_loader)
        print('\nTraining D_Loss: %5.4f, G_Loss: %5.4f' % (d_running_loss, g_running_loss))
        return d_running_loss, g_running_loss

    def test_model(self):
        with torch.no_grad():
            self.D.eval()
            self.G.eval()
            start_time = time.time()
            for batch_idx, (imgs, targets) in enumerate(self.test_loader):
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                imgs_fake = self.G(targets)
                imgs_concat = torch.cat([imgs, imgs_fake], dim=0)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(batch_idx+1))
                save_image(self.denorm(imgs_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Test Iteration: %d/%d, Saved: %s' % (batch_idx+1, len(self.test_loader), result_path), end="\r", flush=True)
            end_time = time.time()
            print('\nTest Completed. Time: %d s' % (end_time - start_time))
