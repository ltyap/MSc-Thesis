import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tabular_dataset import TabularDataset
import os
import numpy as np
import pandas as pd
import torch.distributions as tdists


class CGAN:
    def __init__(self, config, nn_spec, constants) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
        self.disc = nn_spec['disc_spec']['type'](nn_spec['disc_spec']).to(self.device)
        self.gen = nn_spec['gen_spec']['type'](nn_spec['gen_spec']).to(self.device)
        self.config = config
        self.kernel_scale = None
        self.val_ll = []
        self.train_ll = []
        self.epoch_ll = []
        self.discriminator_loss = []
        self.generator_loss = []
        self.nn_spec = nn_spec
        self.epoch_loss = []
        self.fooling = []
        self.constants = constants
        self.plots_path = os.path.join(constants['plot_path'],constants['plt_dataset_name'],constants['file_name'])
        self.param_dir = os.path.join(constants['param_path'],constants['file_name'])

    def train(self, train_data, val_func):
        train_tab = TabularDataset(train_data)
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

        gen_opt = torch.optim.RMSprop(self.gen.parameters(),lr = self.config["gen_lr"])
        disc_opt = torch.optim.RMSprop(self.disc.parameters(), lr = self.config["disc_lr"])
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
                new_gen_batch = self.gen(new_gen_input, data_batch)
                new_gen_logits = self.disc(torch.cat((x_batch,new_gen_batch),dim = 1), data_batch)
                gen_loss = self.gen_loss(new_gen_logits)
                gen_loss.backward()
                gen_opt.step()

                batch_fooling = torch.mean(torch.sigmoid(new_gen_logits))
                epoch_fooling.append(batch_fooling.item())
                epoch_disc_loss.append(disc_loss.item())
                epoch_gen_loss.append(gen_loss.item())

            self.discriminator_loss.append(np.mean(epoch_disc_loss))
            self.generator_loss.append(np.mean(epoch_gen_loss))
            self.fooling.append(np.mean(epoch_fooling))

            if val_func and ((epoch+1) % self.config["val_interval"] == 0):
                evaluation_vals, evaluation_vals_train = val_func(self, epoch)
                tmp = evaluation_vals["ll"]
                tmp2 = evaluation_vals_train["ll"]
                self.val_ll.append(tmp)
                self.train_ll.append(tmp2)
                self.epoch_ll.append(epoch+1)
                if (best_epoch_i==None) or evaluation_vals["ll"] > best_ll:
                    best_ll = evaluation_vals["ll"]
                    best_mae = evaluation_vals["mae"]
                    best_epoch_i = epoch
                    model_params = {
                        "gen": self.gen.state_dict(),
                        "disc": self.disc.state_dict(),
                    }
                    torch.save(model_params, best_save_path)

        self.epoch_disc_loss = epoch_disc_loss
        self.epoch_gen_loss = epoch_gen_loss
        self.epoch_fooling = epoch_fooling

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

        title = 'loss'
        plt.figure()
        plt.plot(self.discriminator_loss, label = 'disc loss')
        plt.plot(self.generator_loss, label = "gen_loss")
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'Fooling'
        plt.figure()
        plt.plot(self.fooling, label='fooling')
        plt.legend()
        plt.title(title)
        images_save_path = os.path.join(self.plots_path,"{}.png".format(title))
        plt.savefig(images_save_path)

        print("best ll:%g",best_ll)
        print("best mae:%g",best_mae)
        print("best epoch:%g",best_epoch_i)
        checkpoint = torch.load(best_save_path, map_location=self.device)
        self.disc.load_state_dict(checkpoint["disc"])
        if "gen" in checkpoint:
            self.gen.load_state_dict(checkpoint["gen"])

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
            samples = (self.gen(gen_input))
        return samples
    def compute_gradient_penalty():
        raise NotImplementedError("Only applicable for WGAN")
    def RandomWeightedAverage():
        raise NotImplementedError("Only applicable for WGAN")
    def logging(self):
        losses = pd.DataFrame(zip(self.epoch_loss,self.discriminator_loss, self.generator_loss),
                              columns=['epoch','discriminator','generator'])
        losses.to_csv('{}/nn_losses.csv'.format(self.plots_path))
        losses = pd.DataFrame(zip(self.epoch_ll,self.train_ll, self.val_ll),
                              columns=['epoch','train','validation'])
        losses.to_csv('{}/ll_losses.csv'.format(self.plots_path))
