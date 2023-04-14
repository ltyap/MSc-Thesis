from cgan import CGAN
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import grad as torch_grad
from tabular_dataset import TabularDataset
import time
import os
import matplotlib.pyplot as plt
import ot

class WCGAN(CGAN):
    def __init__(self, config, nn_spec, constants) -> None:
        self.lambda_gp = config['lambda_gp']
        self.n_critic = config['n_critic']
        self.gp = []
        self.discriminator_loss_val = []
        super().__init__(config, nn_spec, constants)        
        self.critic = self.disc
    def compute_gradient_penalty(self, labels, gen_samples, real_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        batch_size = gen_samples.size()[0]
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((batch_size,1)).to(self.device)
        alpha = alpha.expand_as(gen_samples)
        # Get random interpolation between real and fake samples
        interpolates = alpha * real_samples + ((1 - alpha) * gen_samples)
        # set it to require grad info
        interpolates.requires_grad_(True)
        # Calculate probability of interpolated examples
        d_interpolates = self.critic(torch.cat((labels,interpolates), dim = 1))
        fake = torch.ones_like(d_interpolates)
        # Get gradient w.r.t. interpolates
        gradients = torch_grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size()[0], -1)
        gradients_norm = gradients.norm(2, dim=1)
        if self.config['one-sided']:
            gradient_penalty = (torch.clamp(gradients_norm-1, min=0)**2).mean()
        else:
            gradient_penalty = torch.pow(gradients_norm-1, 2).mean()

        return self.lambda_gp*gradient_penalty

    def gen_loss(self, gen_logits):
        return -torch.mean(gen_logits)
    def disc_loss(self, gen_logits, data_logits, gradient_penalty):
        return torch.mean(gen_logits)-torch.mean(data_logits)+gradient_penalty

    def train(self, train_data, validation_data, val_func):
        start_time = time.time()
        train_tab = TabularDataset(train_data)
        val_tab = TabularDataset(validation_data)
        n_samples = len(train_tab)
        train_loader = torch.utils.data.DataLoader(train_tab,
                                                    batch_size = self.config["batch_size"],
                                                    shuffle = True)
        self.gen.train()
        self.critic.train()

        best_ll = None #best validation log-likelihood
        best_mae = None
        best_epoch_i = None
        if not os.path.exists(self.param_dir):
            os.makedirs(self.param_dir)
        assert os.path.exists(self.param_dir),("model parameter folder {} does not exist".format(self.param_dir))

        best_save_path = os.path.join(self.param_dir,
                            "epoch_best.pt") # Path to save best params to

        # gen_opt = torch.optim.RMSprop(self.gen.parameters(),lr = self.config["gen_lr"])
        # disc_opt = torch.optim.RMSprop(self.critic.parameters(), lr = self.config["disc_lr"])


        gen_opt = torch.optim.Adam(self.gen.parameters(),lr = self.config["gen_lr"],betas=(0.5,0.9))
        disc_opt = torch.optim.Adam(self.critic.parameters(), lr = self.config["disc_lr"],betas=(0.5,0.9))

        one = torch.tensor(1, dtype = torch.float).to(self.device)
        mone = (one * -1).to(self.device)

        for epoch in range(self.config["epochs"]):
            # --------------
            # Train critic
            # --------------
            for iter, (x_batch, data_batch) in enumerate(train_loader):
                for p in self.critic.parameters():
                    p.requires_grad = True
                batch_size = data_batch.shape[0]
                x_batch = x_batch.to(self.device)
                data_batch = data_batch.to(self.device)
            
                self.critic.zero_grad()
                #Train critic on real images
                data_logits = self.critic(torch.cat((x_batch,data_batch), dim = 1))
                data_logits = data_logits.mean()
                data_logits.backward(mone)

                #Sample from generator
                noise_batch = self.get_gaussian().sample([batch_size]).to(self.device)
                gen_input = torch.cat((x_batch, noise_batch), dim = 1)
                with torch.no_grad():
                    gen_output = self.gen(gen_input)
                gen_output = gen_output.detach()
                #Train critic on fake images
                gen_logits = self.critic(torch.cat((x_batch,gen_output), dim = 1))
                gen_logits = gen_logits.mean()
                gen_logits.backward(one)

                #gradient penalty
                gp = self.compute_gradient_penalty(x_batch, gen_output, data_batch.detach())
                gp.backward()

                # disc_loss = self.disc_loss(gen_logits, data_logits, gp)
                disc_loss = gen_logits-data_logits+gp
                # disc_loss.backward()
                disc_opt.step()

                # --------------
                # Train generator
                # --------------
                if ((iter+1)%self.n_critic) == 0:
                    print('training generator')
                    for p in self.critic.parameters():
                        p.requires_grad = False
                    self.gen.zero_grad()
                    # Get new noise vector
                    n_gen_samples = batch_size
                    new_noise_batch = self.get_gaussian().sample([n_gen_samples]).to(self.device)
                    new_gen_batch = self.gen(torch.cat((x_batch,new_noise_batch),dim = 1))
                    new_gen_logits = self.critic(torch.cat((x_batch,new_gen_batch),dim = 1))
                    gen_loss = -new_gen_logits.mean()
                    gen_loss.backward(one)
                    gen_opt.step()

            # disc loss for validation data
            x_batch_val = (val_tab.xs).to(self.device)
            data_batch_val = (val_tab.ys).to(self.device)
            batch_size_val = len(data_batch_val)      

            noise_batch_val = self.get_gaussian().sample([batch_size_val]).to(self.device)
            gen_input_val = torch.cat((x_batch_val, noise_batch_val), dim = 1)
            with torch.no_grad():
                gen_output_val = self.gen(gen_input_val)
            gen_output_val = gen_output_val.detach()
            gen_logits_val = self.critic(torch.cat((x_batch_val,gen_output_val), dim = 1))
            gen_logits_val = gen_logits_val.mean()

            data_logits_val = self.critic(torch.cat((x_batch_val,data_batch_val), dim = 1))
            data_logits_val = data_logits_val.mean()

            gp_val = self.compute_gradient_penalty(x_batch_val, gen_output_val, data_batch_val.detach())
            disc_loss_val = gen_logits_val-data_logits_val+gp_val

            self.discriminator_loss.append(disc_loss.item())
            self.discriminator_loss_val.append(disc_loss_val.item())
            self.gp.append(gp.item())
            self.generator_loss.append(gen_loss.item())
            self.epoch_loss.append(epoch)

            if val_func and ((epoch+1) % self.config["val_interval"] == 0):
                evaluation_vals, evaluation_vals_train = val_func(self, epoch)
                self.val_ll.append(evaluation_vals["ll"])
                self.train_ll.append(evaluation_vals_train["ll"])
                self.epoch_ll.append(epoch+1)
                self.wasserstein_dist.append(evaluation_vals["Wasserstein-2 dist"])
                if (best_epoch_i==None) or evaluation_vals["ll"] > best_ll:
                    best_ll = evaluation_vals["ll"]
                    best_mae = evaluation_vals["mae"]
                    best_epoch_i = epoch
                    model_params = {
                        "gen": self.gen.state_dict(),
                        "disc": self.critic.state_dict(),
                    }
                    torch.save(model_params, best_save_path)


        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

        title = 'history'
        plt.figure()
        plt.plot(self.epoch_ll,self.val_ll, label = "val ll")
        plt.plot(self.epoch_ll,self.train_ll, label = "train ll")
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'generator losses'
        plt.figure()
        plt.plot(self.generator_loss, label = "gen_loss")
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'discriminator losses'
        plt.figure()
        # plt.plot(self.gp, label = 'gradient penalty')
        plt.plot(-np.array(self.discriminator_loss), label = 'disc loss')
        plt.plot(-np.array(self.discriminator_loss_val), label = 'validation disc loss')
        plt.title(title)
        plt.ylabel('Negative critic loss')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'wasserstein distance'
        plt.figure()
        plt.plot(self.epoch_ll,self.wasserstein_dist)
        plt.title(title)
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        print("best ll:{}".format(best_ll))
        print("best mae:{}".format(best_mae))
        print("best epoch:{}".format(best_epoch_i))

        checkpoint = torch.load(best_save_path, map_location=self.device)
        self.critic.load_state_dict(checkpoint["disc"])
        if "gen" in checkpoint:
            self.gen.load_state_dict(checkpoint["gen"])

        self.logging()


        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
    def logging(self):
        losses = pd.DataFrame(zip(self.epoch_ll,self.discriminator_loss, self.gp,self.wasserstein_dist, self.generator_loss),\
                              columns=['epoch','discriminator','gradient penalty','wasserstein distance','generator'])
        losses.to_csv('{}/nn_losses.csv'.format(self.plots_path))
        losses = pd.DataFrame(zip(self.epoch_ll,self.train_ll, self.val_ll),columns=['epoch','train','validation'])
        losses.to_csv('{}/ll_losses.csv'.format(self.plots_path))

# f-CGAN
class PearsonCGAN(CGAN):
    def disc_loss(self, gen_logits, data_logits):
        return torch.mean(-1.*data_logits) +\
            torch.mean(gen_logits*(0.25*gen_logits + 1.0))
    def gen_loss(self, gen_logits):
        return torch.mean(-1*gen_logits)
class LSCGAN(CGAN):
    def disc_loss(self, gen_logits, data_logits):
        return 0.5*(nn.functional.mse_loss(data_logits, torch.ones_like(data_logits))
                + nn.functional.mse_loss(gen_logits, torch.zeros_like(gen_logits)))
    def gen_loss(self, gen_logits):
        return 0.5*nn.functional.mse_loss(gen_logits, torch.ones_like(gen_logits))
class NeymanCGAN(CGAN):
    def disc_loss(self, gen_logits,data_logits):
        return torch.mean(-1.+torch.exp(data_logits)) +\
            torch.mean(2.*(1. - torch.exp(0.5*gen_logits)))
    def gen_loss(self, gen_logits):
        return torch.mean(-1.+torch.exp(gen_logits))
