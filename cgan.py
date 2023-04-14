import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tabular_dataset import TabularDataset
from torch.autograd import grad as torch_grad
import os
import numpy as np
import pandas as pd
import torch.distributions as tdists
import time
import ot

class CGAN:
    def __init__(self, config, nn_spec, constants) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
        self.nn_spec = nn_spec
        self.disc = nn_spec['disc_spec']['type'](nn_spec['disc_spec']).to(self.device)
        self.gen = nn_spec['gen_spec']['type'](nn_spec['gen_spec']).to(self.device)
        self.config = config
        self.kernel_scale = None
        self.val_ll = []
        self.train_ll = []
        self.epoch_ll = []
        self.discriminator_loss = []
        self.generator_loss = []
        self.epoch_loss = []
        self.fooling = []
        self.wasserstein1_dist = []
        self.wasserstein2_dist = []
        self.wasserstein1_dist_train = []
        self.wasserstein2_dist_train = []
        self.gradient_norm_val = []
        self.constants = constants
        self.plots_path = os.path.join(constants['plot_path'],constants['plt_dataset_name'],constants['file_name'])
        self.param_dir = os.path.join(constants['param_path'],constants['file_name'])
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
        d_interpolates = self.disc(torch.cat((labels,interpolates), dim = 1))
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
        return torch.max(gradients_norm)

    def train(self, train_data, validation_data, test_data, val_func):
        start_time = time.time()
        train_tab = TabularDataset(train_data)
        val_tab = TabularDataset(validation_data)

        train_loader = torch.utils.data.DataLoader(train_tab,
                                                    batch_size = self.config["batch_size"],
                                                    shuffle = True,
                                                    num_workers = 4)
        self.gen.train()
        self.disc.train()

        best_mae = None
        best_epoch_i = None
        best_ll = None

        if not os.path.exists(self.param_dir):
            os.makedirs(self.param_dir)
        assert os.path.exists(self.param_dir),("model parameter folder {} does not exist".format(self.param_dir))

        best_save_path = os.path.join(self.param_dir,
                            "epoch_best.pt") # Path to save best params to
        if self.nn_spec['disc_spec']['spectral_normalisation'] == None:
            gen_opt = torch.optim.RMSprop(self.gen.parameters(),lr = self.config["gen_lr"])
            disc_opt = torch.optim.RMSprop(self.disc.parameters(), lr = self.config["disc_lr"])
        else:
            gen_opt = torch.optim.Adam(self.gen.parameters(),lr = self.config["gen_lr"], betas=(0.,0.9))
            disc_opt = torch.optim.Adam(self.disc.parameters(), lr = self.config["disc_lr"], betas=(0.,0.9))

        # print(gen_opt)
        for epoch in range(self.config["epochs"]):
            epoch_disc_loss = []
            epoch_gen_loss = []
            epoch_fooling = []
            for _, (x_batch, data_batch) in enumerate(train_loader):
                batch_size = data_batch.shape[0]

                x_batch = x_batch.to(self.device)
                data_batch = data_batch.to(self.device)

                disc_opt.zero_grad()

                noise_batch = self.get_gaussian().sample([batch_size]).to(self.device)

                #Sample from generator
                gen_input = torch.cat((x_batch, noise_batch), dim = 1)
                gen_output = self.gen(gen_input)
                #Train discriminator
                data_logits = self.disc(torch.cat((x_batch, data_batch), dim = 1))
                gen_logits = self.disc(torch.cat((x_batch,gen_output), dim = 1))
                disc_loss = self.disc_loss(gen_logits, data_logits)
                disc_loss.backward()
                disc_opt.step()

                #train generator
                gen_opt.zero_grad()
                n_gen_samples = batch_size
                new_noise_batch = self.get_gaussian().sample([n_gen_samples]).to(self.device)

                new_gen_input = torch.cat((x_batch,new_noise_batch),dim = 1)
                new_gen_batch = self.gen(new_gen_input)
                new_gen_logits = self.disc(torch.cat((x_batch,new_gen_batch),dim = 1))
                gen_loss = self.gen_loss(new_gen_logits)
                gen_loss.backward()
                gen_opt.step()

                batch_fooling = torch.mean(torch.sigmoid(new_gen_logits))
                epoch_fooling.append(batch_fooling.item())
                epoch_disc_loss.append(disc_loss.item())
                epoch_gen_loss.append(gen_loss.item())

            if val_func and ((epoch+1) % self.config["val_interval"] == 0):
                evaluation_vals, evaluation_vals_train = val_func(self, epoch)
                
                self.discriminator_loss.append(np.mean(epoch_disc_loss))
                self.generator_loss.append(np.mean(epoch_gen_loss))
                self.fooling.append(np.mean(epoch_fooling))
                self.val_ll.append(evaluation_vals["ll"])
                self.train_ll.append(evaluation_vals_train["ll"])
                self.epoch_ll.append(epoch+1)
                self.wasserstein1_dist.append(evaluation_vals["Wasserstein-1 dist"])
                self.wasserstein2_dist.append(evaluation_vals["Wasserstein-2 dist"])
                if "Wasserstein-1 dist train" in evaluation_vals_train.keys():
                    self.wasserstein1_dist_train.append(evaluation_vals_train["Wasserstein-1 dist train"])
                    self.wasserstein2_dist_train.append(evaluation_vals_train["Wasserstein-2 dist train"])
                # disc loss for validation data
                x_batch_val = (val_tab.xs).to(self.device)
                data_batch_val = (val_tab.ys).to(self.device)
                batch_size_val = len(data_batch_val)      

                noise_batch_val = self.get_gaussian().sample([batch_size_val]).to(self.device)
                gen_input_val = torch.cat((x_batch_val, noise_batch_val), dim = 1)
                with torch.no_grad():
                    gen_output_val = self.gen(gen_input_val)
                gen_output_val = gen_output_val.detach()
                gen_logits_val = self.disc(torch.cat((x_batch_val,gen_output_val), dim = 1))
                gen_logits_val = gen_logits_val.mean()

                data_logits_val = self.disc(torch.cat((x_batch_val,data_batch_val), dim = 1))
                data_logits_val = data_logits_val.mean()

                gradient_norm_val = self.compute_gradient_penalty(x_batch_val, gen_output_val, data_batch_val.detach())
                self.gradient_norm_val.append(gradient_norm_val.item())
                if (best_epoch_i==None) or evaluation_vals["ll"] > best_ll:
                    best_ll = evaluation_vals["ll"]
                    best_mae = evaluation_vals["mae"]
                    best_epoch_i = epoch
                    model_params = {
                        "gen": self.gen.state_dict(),
                        "disc": self.disc.state_dict(),
                    }
                    torch.save(model_params, best_save_path)

        # self.epoch_disc_loss = epoch_disc_loss
        # self.epoch_gen_loss = epoch_gen_loss
        # self.epoch_fooling = epoch_fooling

        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

        title = 'history'
        plt.figure()
        plt.plot(self.epoch_ll,self.val_ll, label = "val ll")
        plt.plot(self.epoch_ll,self.train_ll, label = "train ll")
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'loss'
        plt.figure()
        plt.plot(self.epoch_ll,self.discriminator_loss, label = 'disc loss')
        plt.plot(self.epoch_ll,self.generator_loss, label = "gen_loss")
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'Fooling'
        plt.figure()
        plt.plot(self.epoch_ll,self.fooling, label='fooling')
        plt.legend()
        plt.title(title)
        plt.xlabel('Epoch')
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'wasserstein 1 distance'
        plt.figure()
        plt.plot(self.epoch_ll,self.wasserstein1_dist,label = 'Calculated')
        if self.wasserstein1_dist_train:
            plt.plot(self.epoch_ll, self.wasserstein1_dist_train,label = 'Calculated (train)', color = 'r')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'wasserstein 2 distance'
        plt.figure()
        plt.plot(self.epoch_ll,self.wasserstein2_dist,label = 'Calculated')
        if self.wasserstein2_dist_train:
            plt.plot(self.epoch_ll, self.wasserstein2_dist_train,label = 'Calculated (train)', color = 'r')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)


        title = 'gradient norm'
        plt.figure()
        plt.plot(self.epoch_ll,self.gradient_norm_val)
        plt.title(title)
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        print("best ll:",best_ll)
        print("best mae:",best_mae)
        print("best epoch:",best_epoch_i)
        checkpoint = torch.load(best_save_path, map_location=self.device)
        self.disc.load_state_dict(checkpoint["disc"])
        if "gen" in checkpoint:
            self.gen.load_state_dict(checkpoint["gen"])

        calc = self.wasserstein(test_data, gradient_norm_val)
        title = 'W1 estimate'
        plt.figure()
        plt.plot(calc, label = 'calculated')
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        self.logging()

        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
    def wasserstein(self, data, norm):
        '''
            returns W1 distance for each pdf location
        '''
        x = data.x.detach()
        y = data.y.detach()
        x_unique, idx, counts = np.unique(data.x, return_counts = True, return_index = True, axis = 0)
        tmp = np.argsort(idx)
        idx = idx[tmp]
        x_unique = x_unique[tmp]
        counts = counts[tmp]

        start_idx = idx
        end_idx = idx+counts

        tmp = []
        tmp2 = []
        for i, (start, end) in enumerate(zip(start_idx,end_idx)):
            x_real = x[start:end]
            y_real = y[start:end]
            x_unique_repeated = torch.Tensor(x_unique[i,:]).repeat(1000, 1)
            samples = self.sample(x_unique_repeated, batch_size=len(x_unique_repeated))
            
            # calculated
            M_wasserstein1 = ot.wasserstein_1d(samples.cpu().detach().numpy(), y_real.numpy(), p = 1)
            y_real_mean = np.mean(y_real.numpy(), axis=0)
            M_wasserstein1_normalised = np.abs(M_wasserstein1/y_real_mean)
            # if i == 49:
            #     print('x',x_unique_repeated)
            #     print(y_real)
            #     print(y_real_mean)
            #     print(M_wasserstein1_normalised)
            # Estimated from loss function
           

            tmp.append(M_wasserstein1_normalised)
        Wasserstein1_dist = np.array(tmp)

        return Wasserstein1_dist
 
    def gen_loss(self, gen_logits):
        return self.bce_logits(gen_logits, torch.ones_like(gen_logits))
    def disc_loss(self, gen_logits, data_logits):

        disc_loss_gen = self.bce_logits(gen_logits,torch.zeros_like(gen_logits))
        disc_loss_data = self.bce_logits(data_logits, torch.ones_like(data_logits))
        disc_loss = disc_loss_gen+disc_loss_data
        return disc_loss

    def get_gaussian(self):
        return tdists.multivariate_normal.MultivariateNormal(
                torch.zeros(self.config["noise_dim"], device=self.device),
                torch.eye(self.config["noise_dim"], device=self.device)
            ) #isotropic

    @torch.no_grad()
    def eval(self, dataset, kde_eval, use_best_kernel_scale=False):
        ks = None

        if use_best_kernel_scale:
            assert self.kernel_scale, "No kernel scale stored in class CGAN"
            ks = self.kernel_scale
        evaluation_vals, best_kernel_scale = kde_eval(self, dataset, kernel_scale = ks)
        self.kernel_scale = best_kernel_scale
        return evaluation_vals

    @torch.no_grad()
    def sample(self, x, batch_size = None, fixed_noise=False):
        n_samples = x.shape[0]
        x = x.to(self.device)

        noise_sample = self.get_gaussian().sample([n_samples]).to(self.device)

        if batch_size and n_samples > batch_size:
            batch_iterator = zip(torch.split(x, batch_size,dim=0),
                                 torch.split(noise_sample, batch_size, dim=0))
            sample_list = []
            for x_batch, noise_batch in batch_iterator:
                gen_input = torch.cat((x_batch, noise_batch), dim = 1)
                sample_batch = (self.gen(gen_input))
                sample_list.append(sample_batch)
            samples = torch.cat(sample_list, dim = 0)
        else:
            gen_input = torch.cat((x,noise_sample), dim = 1)
            samples = self.gen(gen_input)
        return samples
    def logging(self):
        if self.wasserstein1_dist_train:
            col = ['discriminator','generator', 'fooling', 
                   'Actual wasserstein 1 distance (train)', 'Actual wasserstein 1 distance (val)',
                    'Actual wasserstein 2 distance (train)','Actual wasserstein 2 distance (val)']
            losses = pd.DataFrame(zip(self.discriminator_loss, self.generator_loss, self.fooling,
                                    self.wasserstein1_dist_train, self.wasserstein1_dist,
                                    self.wasserstein2_dist_train, self.wasserstein2_dist),
                                columns=col)
        else:
            col = ['discriminator','generator', 'fooling', 
                   'Actual wasserstein 1 distance (val)', 'Actual wasserstein 2 distance (val)']
            losses = pd.DataFrame(zip(self.discriminator_loss, self.generator_loss, self.fooling,
                                    self.wasserstein1_dist, self.wasserstein2_dist),
                                columns=col)
        losses.to_csv('{}/nn_losses.csv'.format(self.plots_path))
        losses = pd.DataFrame(zip(self.epoch_ll,self.train_ll, self.val_ll),
                              columns=['epoch','train','validation'])
        losses.to_csv('{}/ll_losses.csv'.format(self.plots_path))
