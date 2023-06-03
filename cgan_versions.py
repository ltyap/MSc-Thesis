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
        self.gp = []
        self.discriminator_loss_val = []
        self.estimated_w1_dist = []
        self.estimated_w1_dist_val = []
        self.lambda_gp = config['lambda_gp']
        self.n_critic = config['n_critic']
        self.gradient_norm = []
        self.gradient_norm_val = []

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
        # print('gradient shape',len(gradients))
        gradients = gradients.view(len(gradients), -1)
        # print('gradient shape view',gradients.shape)
        gradients_norm = gradients.norm(2, dim=1)
        # print('gradient norm shape',gradients_norm.shape)

        if self.config['one-sided']:
            gradient_penalty = (torch.clamp(gradients_norm-1, min=0)**2).mean()
        else:
            gradient_penalty = torch.mean((gradients_norm-1)**2)

        return self.lambda_gp*gradient_penalty, torch.max(gradients_norm)

    def gen_loss(self, gen_logits):
        return -torch.mean(gen_logits)
    def disc_loss(self, gen_logits, data_logits, gradient_penalty):
        return torch.mean(gen_logits)-torch.mean(data_logits)+gradient_penalty

    def train(self, train_data, validation_data, test_data, val_func):
        start_time = time.time()
        train_tab = TabularDataset(train_data)
        train_tab_stddev = torch.std(train_tab.ys)
        val_tab = TabularDataset(validation_data)
        val_tab_stddev = torch.std(val_tab.ys)
        train_loader = torch.utils.data.DataLoader(train_tab,
                                                    batch_size = self.config["batch_size"],
                                                    shuffle = True)
        batching = iter(train_loader)
        
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
        print("Best model saved to:", best_save_path)

        # gen_opt = torch.optim.Adam(self.gen.parameters(),lr = self.config["gen_lr"],betas=(0.5,0.9))
        # disc_opt = torch.optim.Adam(self.critic.parameters(), lr = self.config["disc_lr"],betas=(0.5,0.9))
        gen_opt = torch.optim.RMSprop(self.gen.parameters(),lr = self.config["gen_lr"])
        disc_opt = torch.optim.RMSprop(self.critic.parameters(), lr = self.config["disc_lr"])

        for epoch in range(self.config["epochs"]):
            # --------------
            # Train critic
            # --------------
            mean_iteration_gradient_norm = 0
            for p in self.critic.parameters():
                p.requires_grad = True
            for _ in range(self.n_critic):
                try:
                    x_batch, data_batch = next(batching)
                except StopIteration:
                    batching = iter(train_loader)
                    x_batch, data_batch = next(batching)
                batch_size = data_batch.shape[0]
                x_batch = x_batch.to(self.device)
                data_batch = data_batch.to(self.device)
            
                self.critic.zero_grad()

                #Train critic on real images
                data_logits = self.critic(torch.cat((x_batch,data_batch), dim = 1))
                # data_logits.backward(mone)

                #Sample from generator
                noise_batch = self.noise_dist.sample([batch_size]).to(self.device)
                gen_input = torch.cat((x_batch, noise_batch), dim = 1)
                with torch.no_grad():
                    gen_output = self.gen(gen_input)
                gen_output = gen_output.detach()
                #Train critic on fake images
                gen_logits = self.critic(torch.cat((x_batch,gen_output), dim = 1))

                #gradient penalty
                gp, gradient_norm = self.compute_gradient_penalty(x_batch, gen_output, data_batch.detach())
                mean_iteration_gradient_norm += gradient_norm.item() / self.n_critic
                disc_loss = self.disc_loss(gen_logits, data_logits, gp)
                disc_loss.backward(retain_graph=True)
                estimated_w1_dist = (disc_loss - gp)/(gradient_norm*torch.std(data_batch))
                disc_opt.step()
            # --------------
            # Train generator
            # --------------
            for p in self.critic.parameters():
                p.requires_grad = False
            self.gen.zero_grad()
            # Get new gen batch
            n_gen_samples = x_batch.shape[0]
            new_noise_batch = self.noise_dist.sample([n_gen_samples]).to(self.device)
            new_gen_batch = self.gen(torch.cat((x_batch,new_noise_batch),dim = 1))
            new_gen_logits = self.critic(torch.cat((x_batch,new_gen_batch),dim = 1))
            gen_loss = self.gen_loss(new_gen_logits)
            gen_loss.backward()
            gen_opt.step()

            if val_func and ((epoch+1) % self.config["val_interval"] == 0):
                evaluation_vals, evaluation_vals_train = val_func(self, epoch)
                self.val_ll.append(evaluation_vals["ll"])
                self.train_ll.append(evaluation_vals_train["ll"])
                self.epoch_ll.append(epoch+1)
                self.w1_dist.append(evaluation_vals["Wasserstein-1 dist"])
                self.w2_dist.append(evaluation_vals["Wasserstein-2 dist"])
                self.w1_dist_train.append(evaluation_vals_train["Wasserstein-1 dist"])
                self.w2_dist_train.append(evaluation_vals_train["Wasserstein-2 dist"])
                if "mean cond Wasserstein-1 dist" in evaluation_vals_train.keys():
                    self.mean_cond_w1_dist_train.append(evaluation_vals_train['mean cond Wasserstein-1 dist'])
                    self.mean_cond_w2_dist_train.append(evaluation_vals_train['mean cond Wasserstein-2 dist'])
                self.mean_cond_w1_dist.append(evaluation_vals['mean cond Wasserstein-1 dist'])
                self.mean_cond_w2_dist.append(evaluation_vals['mean cond Wasserstein-2 dist'])
                self.gradient_norm.append(mean_iteration_gradient_norm)
                self.discriminator_loss.append(disc_loss.item())
                self.estimated_w1_dist.append(estimated_w1_dist.item())
                self.gp.append(gp.item())
                self.generator_loss.append(gen_loss.item())
                self.epoch_loss.append(epoch)

                # disc loss/estimated w1 dist for validation data
                x_batch_val = (val_tab.xs).to(self.device)
                data_batch_val = (val_tab.ys).to(self.device)
                batch_size_val = len(data_batch_val)      

                noise_batch_val = self.noise_dist.sample([batch_size_val]).to(self.device)
                gen_input_val = torch.cat((x_batch_val, noise_batch_val), dim = 1)
                with torch.no_grad():
                    gen_output_val = self.gen(gen_input_val)
                gen_output_val = gen_output_val.detach()
                gen_logits_val = self.critic(torch.cat((x_batch_val,gen_output_val), dim = 1))
                gen_logits_val = gen_logits_val.mean()
                data_logits_val = self.critic(torch.cat((x_batch_val,data_batch_val), dim = 1))
                data_logits_val = data_logits_val.mean()
                gp_val,gradient_norm_val = self.compute_gradient_penalty(x_batch_val, gen_output_val, data_batch_val.detach())
                disc_loss_val = gen_logits_val-data_logits_val+gp_val

                estimated_w1_dist_val = (disc_loss_val-gp_val)/(gradient_norm_val*val_tab_stddev)
                self.estimated_w1_dist_val.append(estimated_w1_dist_val.item())
                self.discriminator_loss_val.append(disc_loss_val.item())
                self.gradient_norm_val.append(gradient_norm_val.item())

                if (best_epoch_i==None) or evaluation_vals["ll"] > best_ll:
                    best_ll = evaluation_vals["ll"]
                    best_mae = evaluation_vals["mae"]
                    best_epoch_i = epoch
                    model_params = {
                        "gen": self.gen.state_dict(),
                        "disc": self.critic.state_dict(),
                    }
                    self.cond_w1_dist = evaluation_vals['cond Wasserstein-1 dist']
                    self.cond_w2_dist = evaluation_vals["cond Wasserstein-2 dist"]
                    if "cond Wasserstein-1 dist" in evaluation_vals_train.keys():
                        self.cond_w1_dist_train = evaluation_vals_train['cond Wasserstein-1 dist']
                        self.cond_w2_dist_train = evaluation_vals_train['cond Wasserstein-2 dist']
                    torch.save(model_params, best_save_path)
                    # print("saved model")

        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)
        print("Plots path:",self.plots_path)
        title = 'history'
        plt.figure()
        plt.plot(self.epoch_ll,self.val_ll, label = "val ll")
        plt.plot(self.epoch_ll,self.train_ll, label = "train ll")
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'generator losses'
        plt.figure()
        plt.plot(self.epoch_ll,self.generator_loss, label = "gen_loss")
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'discriminator losses'
        plt.figure()
        plt.plot(self.epoch_ll,-np.array(self.discriminator_loss), label = 'disc loss')
        plt.plot(self.epoch_ll,-np.array(self.discriminator_loss_val), label = 'validation disc loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Negative critic loss')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'wasserstein 1 distance'
        plt.figure()
        plt.plot(self.epoch_ll, self.w1_dist,label = 'Calculated (val)', color = 'b')
        plt.plot(self.epoch_ll, self.w1_dist_train,label = 'Calculated (train)', color = 'r')
        plt.plot(self.epoch_ll, -np.array(self.estimated_w1_dist), label='Estimated (train)', color = 'r', linestyle='--')
        plt.plot(self.epoch_ll, -np.array(self.estimated_w1_dist_val), label='Estimated (val)', color = 'b', linestyle='--')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'wasserstein 2 distance'
        plt.figure()
        plt.plot(self.epoch_ll,self.w2_dist,label = 'Calculated (val)', color = 'b')
        plt.plot(self.epoch_ll,self.w2_dist_train,label = 'Calculated (train)', color = 'r')
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'gradient norm'
        plt.figure()
        plt.plot(self.epoch_ll,self.gradient_norm)
        plt.xlabel('Epoch')
        plt.title(title)
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'mean conditional W1 distance'
        plt.figure()
        plt.plot(self.epoch_ll,self.mean_cond_w1_dist, label = 'test')
        if any(self.mean_cond_w1_dist_train):
            plt.plot(self.epoch_ll, self.mean_cond_w1_dist_train, label = 'train')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'mean conditional W2 distance'
        plt.figure()
        plt.plot(self.epoch_ll,self.mean_cond_w2_dist, label = 'test')
        if any(self.mean_cond_w2_dist_train):
            plt.plot(self.epoch_ll, self.mean_cond_w2_dist_train, label = 'train')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'conditional W1 distance'
        plt.figure()
        if any(self.cond_w1_dist_train):
            plt.plot(self.cond_w1_dist_train, label = 'train')
        plt.plot(self.cond_w1_dist, label = 'test')
        plt.xlabel('Index')
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

        title = 'conditional W2 distance'
        plt.figure()
        if any(self.cond_w2_dist_train):
            plt.plot(self.cond_w2_dist_train, label='train')
        plt.plot(self.cond_w2_dist, label='test')
        plt.xlabel('Index')
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)
        plt.close()

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
        if any(self.mean_cond_w1_dist_train):
            col = ['epoch','discriminator','discriminator (val)','generator',
               'Actual wasserstein-1 distance (Training)','Actual Wasserstein-1 distance (Validation)',
                'Estimated wasserstein-1 distance (Training)','Estimated wasserstein-1 distance (Validation)',
                'Actual Wasserstein-2 distance (Training)', "Actual wasserstein-2 distance",
                'Mean Cond. Wasserstein-1 dist (Training)', 'Mean Cond. Wasserstein-1 dist (Test)',
                'Mean Cond. Wasserstein-2 dist (Training)', 'Mean Cond. Wasserstein-2 dist (Test)']
            losses = pd.DataFrame(zip(self.epoch_ll, self.discriminator_loss, self.discriminator_loss_val,self.generator_loss,
                                self.w1_dist_train, self.w1_dist,
                                self.estimated_w1_dist, self.estimated_w1_dist_val,
                                self.w2_dist_train, self.w2_dist,
                                self.mean_cond_w1_dist_train,self.mean_cond_w1_dist,
                                self.mean_cond_w2_dist_train, self.mean_cond_w2_dist),
                              columns=col)
            cond_wdist = pd.DataFrame(zip(self.cond_w1_dist_train, self.cond_w1_dist,
                                self.cond_w2_dist_train, self.cond_w2_dist),
                                columns=["cond w1 dist (train)","cond w1 dist (test)",
                                        "cond w2 dist (train)","cond w2 dist (test)"])
        else:
            col = ['epoch','discriminator','discriminator (val)','generator',
                'Actual wasserstein-1 distance (Validation)',
                'Estimated Wasserstein-1 distance (Training)','Estimated wasserstein-1 distance (Validation)',
                'Actual Wasserstein-2 distance',
                'Mean Cond. Wasserstein-1 dist (Test)','Mean Cond. Wasserstein-2 dist (Test)']
            losses = pd.DataFrame(zip(self.epoch_ll,self.discriminator_loss, self.discriminator_loss_val,self.generator_loss,
                                    self.w1_dist,
                                    self.estimated_w1_dist, self.estimated_w1_dist_val,
                                    self.w2_dist,
                                    self.mean_cond_w1_dist, self.mean_cond_w2_dist),
                                columns=col)
            cond_wdist = pd.DataFrame(zip(self.cond_w1_dist, self.cond_w2_dist),
                                columns=["cond w1 dist (test)", "cond w2 dist (test)"])

        losses.to_csv('{}/nn_losses.csv'.format(self.plots_path))
        cond_wdist.to_csv('{}/wdist.csv'.format(self.plots_path))

        ll = pd.DataFrame(zip(self.epoch_ll,self.train_ll, self.val_ll),columns=['epoch','train','validation'])
        ll.to_csv('{}/ll_losses.csv'.format(self.plots_path))

class WdivCGAN(CGAN):
    def __init__(self, config, nn_spec, constants) -> None:
        self.gp = []
        self.discriminator_loss_val = []
        self.lambda_div = config['lambda_div']
        self.n_critic = config['n_critic']
        self.gradient_norm = []
        self.gradient_norm_val = []
        self.power = config['power']
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
        # print('gradient shape',len(gradients))
        gradients = gradients.view(len(gradients), -1)
        # print('gradient shape view',gradients.shape)
        gradients_norm = gradients.norm(2, dim=1)
        # print('gradient norm shape',gradients_norm.shape)

        gradient_penalty = torch.mean((gradients_norm)**self.power)

        return self.lambda_div*gradient_penalty, torch.max(gradients_norm)

    def gen_loss(self, gen_logits):
        return torch.mean(gen_logits)
    def disc_loss(self, gen_logits, data_logits, gradient_penalty):
        return torch.mean(data_logits)-torch.mean(gen_logits)+gradient_penalty

    def train(self, train_data, validation_data, val_func):
        start_time = time.time()
        train_tab = TabularDataset(train_data)
        val_tab = TabularDataset(validation_data)
        train_loader = torch.utils.data.DataLoader(train_tab,
                                                    batch_size = self.config["batch_size"],
                                                    shuffle = True)
        batching = iter(train_loader)
        
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

 
        # gen_opt = torch.optim.Adam(self.gen.parameters(),lr = self.config["gen_lr"],betas=(0.5,0.9))
        # disc_opt = torch.optim.Adam(self.critic.parameters(), lr = self.config["disc_lr"],betas=(0.5,0.9))
        gen_opt = torch.optim.RMSprop(self.gen.parameters(),lr = self.config["gen_lr"])
        disc_opt = torch.optim.RMSprop(self.critic.parameters(), lr = self.config["disc_lr"])

        for epoch in range(self.config["epochs"]):
            # --------------
            # Train critic
            # --------------
            mean_iteration_gradient_norm = 0
            for p in self.critic.parameters():
                p.requires_grad = True
            for _ in range(self.n_critic):
                try:
                    x_batch, data_batch = next(batching)
                except StopIteration:
                    batching = iter(train_loader)
                    x_batch, data_batch = next(batching)
                batch_size = data_batch.shape[0]
                x_batch = x_batch.to(self.device)
                data_batch = data_batch.to(self.device)
            
                self.critic.zero_grad()

                #Train critic on real images
                data_logits = self.critic(torch.cat((x_batch,data_batch), dim = 1))

                #Sample from generator
                noise_batch = self.noise_dist.sample([batch_size]).to(self.device)
                gen_input = torch.cat((x_batch, noise_batch), dim = 1)
                with torch.no_grad():
                    gen_output = self.gen(gen_input)
                gen_output = gen_output.detach()
                #Train critic on fake images
                gen_logits = self.critic(torch.cat((x_batch,gen_output), dim = 1))

                #gradient penalty
                gp, gradient_norm = self.compute_gradient_penalty(x_batch, gen_output, data_batch.detach())
                mean_iteration_gradient_norm += gradient_norm.item() / self.n_critic
                disc_loss = self.disc_loss(gen_logits, data_logits, gp)
                disc_loss.backward(retain_graph=True)
                disc_opt.step()
            self.gradient_norm.append(mean_iteration_gradient_norm)
            # --------------
            # Train generator
            # --------------
            for p in self.critic.parameters():
                p.requires_grad = False
            self.gen.zero_grad()
            # Get new gen batch
            n_gen_samples = x_batch.shape[0]
            new_noise_batch = self.noise_dist.sample([n_gen_samples]).to(self.device)
            new_gen_batch = self.gen(torch.cat((x_batch,new_noise_batch),dim = 1))
            new_gen_logits = self.critic(torch.cat((x_batch,new_gen_batch),dim = 1))
            gen_loss = self.gen_loss(new_gen_logits)
            # gen_loss = -new_gen_logits.mean()
            gen_loss.backward()
            gen_opt.step()

            # disc loss for validation data
            x_batch_val = (val_tab.xs).to(self.device)
            data_batch_val = (val_tab.ys).to(self.device)
            batch_size_val = len(data_batch_val)      

            noise_batch_val = self.noise_dist.sample([batch_size_val]).to(self.device)
            gen_input_val = torch.cat((x_batch_val, noise_batch_val), dim = 1)
            with torch.no_grad():
                gen_output_val = self.gen(gen_input_val)
            gen_output_val = gen_output_val.detach()
            gen_logits_val = self.critic(torch.cat((x_batch_val,gen_output_val), dim = 1))
            gen_logits_val = gen_logits_val.mean()

            data_logits_val = self.critic(torch.cat((x_batch_val,data_batch_val), dim = 1))
            data_logits_val = data_logits_val.mean()

            gp_val,gradient_norm_val = self.compute_gradient_penalty(x_batch_val, gen_output_val, data_batch_val.detach())
            self.gradient_norm_val.append(gradient_norm_val.item())

            disc_loss_val = -gen_logits_val+data_logits_val+gp_val

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
                self.wasserstein_dist.append(evaluation_vals["Wasserstein-1 dist"])

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

        title = 'gradient norm'
        plt.figure()
        plt.plot(self.gradient_norm)
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
        losses = pd.DataFrame(zip(self.epoch_ll,self.discriminator_loss,self.wasserstein_dist, self.generator_loss),\
                              columns=['epoch','discriminator','wasserstein distance','generator'])
        losses.to_csv('{}/nn_losses.csv'.format(self.plots_path))
        losses = pd.DataFrame(zip(self.epoch_ll,self.train_ll, self.val_ll),columns=['epoch','train','validation'])
        losses.to_csv('{}/ll_losses.csv'.format(self.plots_path))




# f-CGAN
# Reverse Kullback-Leibler
class RKLCGAN(CGAN):
    def disc_loss(self, gen_logits, data_logits):
        return torch.mean(torch.exp(data_logits)) +\
            torch.mean(-gen_logits-1.)

    def gen_loss(self, gen_logits):
        return torch.mean(torch.exp(gen_logits))
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
# Kullback-Leibler
# Gradient clipping would likely be useful
class KLCGAN(CGAN):
    def disc_loss(self, gen_logits, data_logits):
        return torch.mean(-1.*data_logits) +\
            torch.mean(torch.exp(gen_logits - 1.))

    def gen_loss(self, gen_logits):
        return torch.mean(-1.*gen_logits)