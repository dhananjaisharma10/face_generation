import os
import time
import torch
import config
import torch.nn as nn
import torch.optim as optim
from utils import plot_images
from dataset import get_loader
import torch.nn.functional as F
import torchvision.utils as vutils
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
    def __init__(self, args):
        super(Runner, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = Generator(in_dim=config.g_input_dim, conv_dim=config.g_conv_dim,
                            out_dim=config.g_out_channels)
        self.D = Discriminator(in_dim=config.d_in_channels, conv_dim=config.d_conv_dim,
                                label_dim=config.d_cls_dim)
        if args.reload_model:
            # Load pre-trained model.
            if args.model_name is None:
                raise ValueError('Requested reload_model but get model_name as None.')
            else:
                model_path = os.path.join('{}/{}'.format(config.models_dir, args.run_id),
                                            args.model_name)
                model_dict = torch.load(model_path, map_location=self.device)
                self.G.load_state_dict(model_dict['Generator'])
                self.D.load_state_dict(model_dict['Discriminator'])
                print('Loaded model:', model_path)
        else:
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

        #WARNING: If we are loading pre trained modals in between epoches to continue training, we should also load the optimiser.

        self.g_optimizer = optim.Adam(self.G.parameters(), lr=config.g_lr, weight_decay=config.g_wd)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=config.d_lr, weight_decay=config.d_wd)
        self.d_update_ratio = config.d_update_ratio
        self.result_dir = config.result_dir
        self.label_dim = config.g_input_dim
        self.adv_loss = nn.BCELoss()

        # A batch of test images
        _, self.fixed_feats = next(iter(self.test_loader))
        self.fixed_feats = self.fixed_feats.view(self.fixed_feats.size(0),
                            self.fixed_feats.size(1), 1, 1).to(self.device)

    def cls_loss(self, preds, targets):
        # return F.binary_cross_entropy_with_logits(preds, targets, reduction='mean') / targets.size(0)
        # FIXME: Why does STARGAN divide by targets.size(0) if reduction is already mean?
        return F.binary_cross_entropy_with_logits(preds, targets, reduction='mean')

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train_model(self):
        self.D.train()
        self.G.train()
        # Progress tracking parameters.
        d_running_loss = 0      # Discriminator running loss.
        g_running_loss = 0      # Generator running loss.
        running_p_real = 0      # Accumulated predicted probability of real batch.
        running_p_fake = 0      # Accumulated predicted probability of fake batch.
        running_p_fake_2 = 0    # Accumulated predicted probability of fake batch after one D update.

        start_time = time.time()
        for itr, (imgs, labels) in enumerate(self.train_loader, 0):
            batch_size = labels.size(0)
            labels, imgs = labels.to(self.device), imgs.to(self.device)
            labels = labels.view(batch_size, self.label_dim, 1, 1)
            fake_imgs = self.G(labels)      # Generate fake images using G.

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            self.D.zero_grad()

            # Train with all-real batch.
            gt_prob = torch.full((batch_size,), config.real_prob, device=self.device)           # Ground truth probabilities.
            pred_prob, pred_cls = self.D(imgs)                                                  # Forward pass real batch through D.
            d_adv_loss = self.adv_loss(pred_prob, gt_prob)                                      # Adversarial loss.
            d_cls_loss = self.cls_loss(pred_cls, labels.view(labels.size(0), labels.size(1)))   # Classification loss.
            d_loss = d_adv_loss + d_cls_loss
            d_loss.backward()                           # Calculate gradients for D.
            running_p_real += pred_prob.mean().item()   # Accumulate predicted probability for real batch.
            d_running_loss += d_loss.item()             # Accumulate loss for D.

            # Train with all-fake batch
            gt_prob.fill_(config.fake_prob)                                                     # Ground truth probabilities.
            pred_prob, pred_cls = self.D(fake_imgs.detach())                                    # Forward pass fake batch through D.
            d_adv_loss = self.adv_loss(pred_prob, gt_prob)                                      # Adversarial loss.
            d_cls_loss = self.cls_loss(pred_cls, labels.view(labels.size(0), labels.size(1)))   # Classification loss.
            d_loss = d_adv_loss + d_cls_loss
            d_loss.backward()                           # Calculate gradients for D.
            running_p_fake += pred_prob.mean().item()   # Accumulate predicted probability for real batch.
            d_running_loss += d_loss.item()             # Accumulate loss for D.

            # Update D
            self.d_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            self.G.zero_grad()

            # For generator loss, we want discriminator to ideally classify
            # fake image as real. So ground truth label will be same as real label.
            gt_prob.fill_(config.real_prob)                                                     # Ground truth probabilities.
            pred_prob, pred_cls = self.D(fake_imgs)                                             # Forward pass fake batch through D.
            g_adv_loss = self.adv_loss(pred_prob, gt_prob)                                      # Adversarial loss.
            g_cls_loss = self.cls_loss(pred_cls, labels.view(labels.size(0), labels.size(1)))   # Classification loss.
            g_l1_loss = torch.mean(torch.abs(imgs - fake_imgs))                                 # L1 loss for fake image.
            g_loss = g_adv_loss + g_cls_loss + g_l1_loss
            g_loss.backward()                           # Calculate gradients for G.
            running_p_fake_2 += pred_prob.mean().item() # Accumulate predicted probability for real batch.
            g_running_loss += g_loss.item()             # Accumulate loss for G.

            # Update G
            self.g_optimizer.step()

            # Cleanup
            torch.cuda.empty_cache()
            del labels, imgs, g_loss, d_loss, pred_prob, pred_cls
            del g_adv_loss, g_cls_loss, g_l1_loss, d_adv_loss, d_cls_loss

            print("Iter: {}/{} | D-Loss: {:.4f} | G-Loss: {:.4f}".format(\
                    itr, len(self.train_loader),(d_running_loss/(itr+1)), (g_running_loss/(itr+1))) + \
                    " | D(x): {:.4f} | D(G(z)): {:.4f} | Post-D(G(z)): {:.4f}".format(\
                    (running_p_real/(itr+1)), (running_p_fake/(itr+1)), (running_p_fake_2/(itr+1))),
                    end="\r", flush=True)

        end_time = time.time()
        min_time = (end_time - start_time) // 60
        sec_time = (end_time - start_time) - (min_time*60)
        print('\nRuntime: {}m{}s'.format(int(min_time), int(sec_time)), flush=True)

        # For plotting
        with torch.no_grad():
            fake_imgs = self.G(self.fixed_feats).detach().cpu()
            result = vutils.make_grid(fake_imgs, padding=2, normalize=True)

        d_running_loss /= len(self.train_loader)
        g_running_loss /= len(self.train_loader)
        return d_running_loss, g_running_loss, result

    def test_model(self):
        with torch.no_grad():
            #self.G.eval()  #FIXME: Why is .eval causing the image to be black/all_zero?
            start_time = time.time()
            for j in range(self.label_dim):
                feat = self.fixed_feats.clone()
                for i in range(feat.size(0)):
                    feat[i,j,:,:] ^= 1
                    # if feat[i,j,:,:] == 0:
                    #     feat[i,j,:,:] = 1
                    # else:
                    #     feat[i,j,:,:] = 0
                fake = self.G(feat).detach().cpu()
                fake = F.upsample(fake, size_new=(2*config.image_size,2*config.image_size), mode='bilinear')
                result = vutils.make_grid(fake, padding=2, normalize=True)
                plot_images(str(j+1), result, self.args.run_id, mode='test')
            end_time = time.time()
            min_time = (end_time - start_time)//60
            sec_time = (end_time - start_time) - (min_time*60)
            print('Test Completed. Time: %dm%ds' % (min_time,sec_time))
            return result
