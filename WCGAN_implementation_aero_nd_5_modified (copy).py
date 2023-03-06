import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import collections
import torch.distributions as tdists
import math
from misc_funcs import indexes, get_samples
from tabular_dataset import TabularDataset
import seaborn as sns
from torch.autograd import grad as torch_grad
import ot
import pandas as pd
import time
plt.rcParams.update({'font.size': 13})
plt.rc("axes", labelsize=18)
plt.rc("axes", titlesize=21)
plt.rc("legend", fontsize=18)

LabelledData = collections.namedtuple("LabelledData",["x","y"])

#import data
DATASET_PATH = './datasets/aero/'
DATASET_NAME = 'TwrBsMyt_ST_DEL'
CHANNEL_NAME = "TwrBsMyt_[kN-m] ST_DEL"

# For saving plots
PLOT_PATH = './plots'
PLT_DATASET_NAME = 'aero_TwrBsMyt_ST_DEL'

# path for saving parameters of model
PARAM_PATH = './param_best'
FILE_NAME = 'aero_wcgan_nd_5_modified'

#CHANGE DIMENSIONS OF DATA ACCORDINGLY
X_DIM = 3
Y_DIM = 1
num_samples_real = 300
dataset_dir = os.path.join(DATASET_PATH,DATASET_NAME)
assert os.path.exists(dataset_dir),("dataset folder {} does not exist".format(dataset_dir))

splits = {}
scatter_plot = 0

for split in ("train","test","val"):
    data_path = os.path.join(dataset_dir,"{}.csv".format(split))
    assert os.path.exists(data_path),"data file {} does not exist".format(data_path)

    data = np.genfromtxt(data_path,delimiter=",")
    if scatter_plot:
        plt.figure
        plt.scatter(data[:,:1],data[:,1:], c='k')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.title(split)
        plt.show()
    torch_data = torch.tensor(data, device="cpu").float()
    splits[split] = LabelledData(x=torch_data[:,:X_DIM],y=torch_data[:,X_DIM:])

train_data = splits["train"]
val_data = splits['val']
test_data = splits['test']

#create NN classes for discrimator/generator
def ffsection(x_dim, other_dim, layer_list, extra_in_hidden_dim=0):
    if not layer_list:
        layers = []
        out_dim = x_dim+extra_in_hidden_dim
    else:
        layers = [nn.Linear(x_dim + other_dim, layer_list[0])]+\
            [nn.Linear(from_dim+extra_in_hidden_dim, to_dim) \
            for from_dim, to_dim in zip(layer_list[:-1], layer_list[1:])]
    out_dim = layer_list[-1]
    return nn.ModuleList(layers), out_dim

class NoiseInjection(nn.Module):
    def __init__(self, nn_spec) -> None:
        super().__init__()
        self.activation = nn_spec["activation"]#relu
        self.nodes_per_layer = nn_spec["nodes_per_layer"]#[64, 64, 64, 64, 64, 64]
        self.other_dim = nn_spec["other_dim"]#5,#noise dimensions
        self.cond_dim = nn_spec["cond_dim"]# 1 #X_DIM
        self.output_dim = nn_spec["output_dim"]#1 #Y_DIM
        self.activation_final = nn_spec['activation_final'] if nn_spec['activation_final'] else None

        self.layers, self.last_layer_dim = ffsection(
            self.cond_dim, self.other_dim, self.nodes_per_layer, self.other_dim)
        self.output_layer = nn.Linear((self.last_layer_dim+self.other_dim),self.output_dim)

    def forward(self, x):
        hidden_repr = x[:,:self.cond_dim]
        noise = x[:,self.cond_dim:]
        for layer in self.layers:
            combined_repr = torch.cat((hidden_repr, noise), dim = 1)
            hidden_repr = self.activation(layer(combined_repr))
            if self.dropout:
                    hidden_repr = self.dropout(hidden_repr)
        hidden_repr = torch.cat((hidden_repr, noise),dim =1)
        if self.activation_final:
            return self.activation_final(self.output_layer(hidden_repr))
        return self.output_layer(hidden_repr)
class FeedForward(nn.Module):
    def __init__(self, nn_spec) -> None:
        super().__init__()
        self.activation = nn_spec["activation"]
        self.nodes_per_layer = nn_spec["nodes_per_layer"]
        self.other_dim = nn_spec["other_dim"]
        self.cond_dim = nn_spec["cond_dim"]
        self.output_dim = nn_spec["output_dim"]
        self.activation_final = nn_spec['activation_final'] if nn_spec['activation_final'] else None

        self.layers, self.last_layer_dim = ffsection(
            self.cond_dim, self.other_dim, self.nodes_per_layer)
        self.output_layer = nn.Linear(self.last_layer_dim,self.output_dim)
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.activation_final:
            return self.activation_final(self.output_layer(x))
        return self.output_layer(x)#no activation fnc before output
class DoubleInputNetwork(nn.Module):
    def __init__(self, nn_spec) -> None:
        super().__init__()

        self.activation = nn_spec["activation"]
        self.dim_x = nn_spec["cond_dim"]

        self.cond_layers, cond_dim = ffsection(nn_spec["cond_dim"], other_dim= 0,
                                               layer_list= nn_spec["cond_layers"])
        self.other_layers, other_dim = ffsection(x_dim = 0,other_dim = nn_spec["other_dim"],
                                                 layer_list=nn_spec["other_layers"])
        self.hidden_layers, hidden_dim = ffsection(cond_dim, other_dim,
                                                    nn_spec["nodes_per_layer"])
        self.output_layer = nn.Linear(hidden_dim, nn_spec["output_dim"])

    def forward(self, x):
        cond_repr = x[:,:self.dim_x]
        other_repr = x[:,self.dim_x:]

        #conditional input
        for layer in self.cond_layers:
            cond_repr = self.activation(layer(cond_repr))
        #other (noise/real data)
        for layer in self.other_layers:
            other_repr = self.activation(layer(other_repr))

        hidden_input = torch.cat((cond_repr, other_repr), dim = 1)
        for layer in self.hidden_layers:
            hidden_input = self.activation(layer(hidden_input))

        output = self.output_layer(hidden_input)
        return output


class CGAN:
    def __init__(self, config, nn_spec) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
        self.disc = nn_spec['critic_spec']['type'](nn_spec['critic_spec']).to(self.device)
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

        param_dir = os.path.join(PARAM_PATH,FILE_NAME)
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        assert os.path.exists(param_dir),("model parameter folder {} does not exist".format(param_dir))

        best_save_path = os.path.join(param_dir,
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
                new_gen_batch = self.gen(new_gen_input)
                new_gen_logits = self.disc(torch.cat((x_batch,new_gen_batch),dim = 1))
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

        if not os.path.exists(os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME)):
            os.makedirs(os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME))

        title = 'history'
        plt.figure()
        plt.plot(self.epoch_ll,self.val_ll, label = "val ll")
        plt.plot(self.epoch_ll,self.train_ll, label = "train ll")
        plt.title(title)
        plt.legend()

        images_save_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'loss'
        plt.figure()
        plt.plot(self.discriminator_loss, label = 'disc loss')
        plt.plot(self.generator_loss, label = "gen_loss")
        plt.title(title)
        plt.legend()

        images_save_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'Fooling'
        plt.figure()
        plt.plot(self.fooling, label='fooling')
        plt.legend()
        plt.title(title)
        images_save_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,"{}.png".format(title))
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
        losses = pd.DataFrame(zip(self.epoch_loss,self.discriminator_loss, self.generator_loss),columns=['epoch','discriminator','generator'])
        losses.to_csv('{}/{}/{}/nn_losses.csv'.format(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME))
        losses = pd.DataFrame(zip(self.epoch_ll,self.train_ll, self.val_ll),columns=['epoch','train','validation'])
        losses.to_csv('{}/{}/{}/ll_losses.csv'.format(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME))

class WCGAN(CGAN):
    def __init__(self, config, nn_spec) -> None:
        super().__init__(config, nn_spec)
        self.lambda_gp = config['lambda_gp']
        self.n_critic = config['n_critic']
        self.critic = self.disc
        self.gp = []
        self.wasserstein_dist = []
        self.discriminator_loss_val = []
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
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.pow(gradients.norm(2,dim=1)-1, 2).mean()

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
                                                    shuffle = True,
                                                    num_workers = 4)
        self.gen.train()
        self.critic.train()

        best_ll = None #best validation log-likelihood
        best_mae = None
        best_epoch_i = None

        param_dir = os.path.join(PARAM_PATH,FILE_NAME)
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        assert os.path.exists(param_dir),("model parameter folder {} does not exist".format(param_dir))

        best_save_path = os.path.join(param_dir,
                            "epoch_best.pt") # Path to save best params to

        # gen_opt = torch.optim.RMSprop(self.gen.parameters(),lr = self.config["gen_lr"])
        # disc_opt = torch.optim.RMSprop(self.critic.parameters(), lr = self.config["disc_lr"])


        gen_opt = torch.optim.Adam(self.gen.parameters(),lr = self.config["gen_lr"],betas=(0.5,0.9))
        disc_opt = torch.optim.Adam(self.critic.parameters(), lr = self.config["disc_lr"],betas=(0.5,0.9))

        one = torch.tensor(1, dtype = torch.float).to(self.device)
        mone = (one * -1).to(self.device)

        for epoch in range(self.config["epochs"]):
            for n_batch, (x_batch, data_batch) in enumerate(train_loader):
                # --------------
                # Train critic
                # --------------
                batch_size = data_batch.shape[0]
                x_batch = x_batch.to(self.device)
                data_batch = data_batch.to(self.device)
                for p in self.critic.parameters():
                    p.requires_grad = True
                # for _ in range(self.n_critic):
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
                wasserstein_dist = data_logits-gen_logits
                # disc_loss.backward()
                disc_opt.step()

                if n_batch % self.n_critic == 0:
                    # --------------
                    # Train generator
                    # --------------
                    for p in self.critic.parameters():
                        p.requires_grad = False
                    self.gen.zero_grad()
                    # Get new gen batch
                    n_gen_samples = batch_size
                    new_noise_batch = self.get_gaussian().sample([n_gen_samples]).to(self.device)
                    new_gen_input = torch.cat((x_batch,new_noise_batch),dim = 1)
                    new_gen_batch = self.gen(new_gen_input)
                    new_gen_logits = self.critic(torch.cat((x_batch,new_gen_batch),dim = 1))
                    new_gen_logits = new_gen_logits.mean()
                    # gen_loss = self.gen_loss(new_gen_logits)
                    new_gen_logits.backward(mone)
                    gen_loss = -new_gen_logits
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
            self.wasserstein_dist.append(wasserstein_dist.item())

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
                        "disc": self.critic.state_dict(),
                    }
                    torch.save(model_params, best_save_path)


        if not os.path.exists(os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME)):
            os.makedirs(os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME))

        title = 'history'
        plt.figure()
        plt.plot(self.epoch_ll,self.val_ll, label = "val ll")
        plt.plot(self.epoch_ll,self.train_ll, label = "train ll")
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'generator losses'
        plt.figure()
        plt.plot(self.generator_loss, label = "gen_loss")
        plt.title(title)
        plt.legend()
        images_save_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'discriminator losses'
        plt.figure()
        # plt.plot(self.gp, label = 'gradient penalty')
        plt.plot(-np.array(self.discriminator_loss), label = 'disc loss')
        plt.plot(-np.array(self.discriminator_loss_val), label = 'validation disc loss')
        plt.title(title)
        plt.ylabel('Negative critic loss')
        plt.legend()
        images_save_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,"{}.png".format(title))
        plt.savefig(images_save_path)

        title = 'wasserstein distance'
        plt.figure()
        plt.plot(self.wasserstein_dist)
        plt.title(title)
        images_save_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,"{}.png".format(title))
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
        losses.to_csv('{}/{}/{}/nn_losses.csv'.format(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME))
        losses = pd.DataFrame(zip(self.epoch_ll,self.train_ll, self.val_ll),columns=['epoch','train','validation'])
        losses.to_csv('{}/{}/{}/ll_losses.csv'.format(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME))


# In[7]:


def evaluate_model(model, data, data_train,data_test, epoch=None, make_plots = True):
    testing = (epoch==None) #returns False/0 if epoch is not None
    config = model.config
    evaluation_vals = model.eval(data, kde_eval, use_best_kernel_scale=testing)
    val_method = "true"
    metric_string = "\t".join(["{}: {:.5}".format(key, validation_val) for
        (key,validation_val) in evaluation_vals.items()])
    print("Epoch {}, {}\t{}".format(epoch, val_method, metric_string))

    tmp = data.x.shape[0]
    data_train_partial = LabelledData(x=data_train.x[:tmp], y=data_train.y[:tmp])
    evaluation_vals_train = model.eval(data_train_partial, kde_eval, use_best_kernel_scale = testing)
    metric_string = "\t".join(["{}: {:.5}".format(key, training_val) for
        (key,training_val) in evaluation_vals_train.items()])
    # print("Training Epoch {}, {}\t{}".format(epoch, val_method, metric_string))

    if make_plots:
        if config["scatter"]:
            model_samples = model.sample(data.x, batch_size = config["eval_batch_size"])
            if type(model_samples) == torch.Tensor:
                model_samples = model_samples.to("cpu")
            model_data = LabelledData(x=data.x, y = model_samples)
            plot_title = 'Training epoch {}'.format(epoch)
            sample_sets = [data, model_data]
            labels = ["Ground Truth", "model"]
            plot_samples(sample_sets, file_name = epoch,
                        labels = labels, title=plot_title, range_dataset=data)
        if config["pdf_index"]:
            opt_value = config["pdf_index"]
            if opt_value:
                list_of_indexes = [int(s) for s in
                            opt_value.split(",")]
                for idx in list_of_indexes:
                    tmp = indexes(test_data.x[idx], test_data.x)
                    model_samples = model.sample(test_data.x[tmp])
                    if type(model_samples) == torch.Tensor:
                        model_samples = model_samples.to("cpu")
                    plt.figure()
                    sns.kdeplot(model_samples.squeeze(), color='b', label='Gen')
                    sns.kdeplot(data_test.y[tmp].squeeze(), color='k', linestyle='--', label='True')
                    title = "Epoch {}, x = {}".format(epoch,test_data.x[idx].to('cpu'))
                    plt.title(title, fontsize=10)
                    plt.legend()
                    savepath = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'learning_prog_idx{}'.format(idx))
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    assert os.path.exists(savepath),("dataset folder {} does not exist".format(savepath))
                    plt.savefig(os.path.join(savepath,'{}.png'.format(epoch)))
                    plt.close()

    return evaluation_vals, evaluation_vals_train

def mae_from_samples(samples, y):
    medians,_ = torch.median(samples, dim=1)
    abs_errors = torch.sum(torch.abs(medians-y), dim = 1)
    return torch.mean(abs_errors).item()

def kde_eval(model, data, kernel_scale=None):
    x = data.x.to(model.device)
    x_repeated = torch.repeat_interleave(x, repeats = model.config["eval_samples"], dim = 0)
    shape_y = data.y.shape
    samples = model.sample(x_repeated, batch_size = model.config["eval_batch_size"]).reshape(
        shape_y[0],model.config["eval_samples"],shape_y[1])
    y = data.y.to(model.device)
    diff = samples - y.unsqueeze(1)
    squarednorm = torch.sum(torch.pow(diff,2), dim = 2)

    if not (kernel_scale == None):
        n_h = 1
        h_squared = torch.tensor([kernel_scale], device = model.device)
    else:
        n_h = model.config["kernel_scales"]
        h_squared = torch.logspace(
            math.log10(model.config["kernel_scale_min"]),
            math.log10(model.config["kernel_scale_max"]), steps = n_h)
    h_squared = h_squared.to(model.device)
    scale_lls = []
    #Batch over kernel scales
    h_squared_splits = torch.split(h_squared, model.config["kde_batch_size"], dim = 0)

    for splits in h_squared_splits:
        normalisation = torch.pow(torch.rsqrt(2.*math.pi*splits), Y_DIM)
        ratio = squarednorm.unsqueeze(dim=2).repeat((1,1,splits.shape[0]))/splits

        ll_y = torch.log(normalisation)+torch.logsumexp(-0.5*(ratio), dim = 1)-\
                math.log(model.config["eval_samples"])
        mean_lls = torch.mean(ll_y, dim = 0)
        scale_lls.append(mean_lls)
    joined_scale_lls = torch.cat(scale_lls, dim = 0)

    argmax = torch.argmax(joined_scale_lls)
    best_scale = h_squared[argmax]

    best_ll = joined_scale_lls[argmax].item()
    best_mae = mae_from_samples(samples, y)
    evaluation_vals = {
        "ll": best_ll,
        "mae": best_mae
    }

    return evaluation_vals, best_scale

def val_func(model, epoch):
    return evaluate_model(model, data = val_data, data_train=train_data,data_test=test_data, epoch = epoch)

def plot_samples(sample_sets, file_name, labels=None, range_dataset=None,
        title=None, both_y=False):
    n_sets = len(sample_sets)
    fig, axes = plt.subplots(nrows=1, ncols=n_sets, squeeze=False,
            figsize=(7.*n_sets, 4.5))
    x_range = (min(range_dataset.x), max(range_dataset.x))
    y_range = (min(range_dataset.y), max(range_dataset.y))
    for set_i, (samples, ax) in enumerate(zip(sample_sets, axes[0])):
        ax.scatter(samples.x, samples.y, s=3, color="green")

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        if labels:
            ax.set(title=labels[set_i])
        if both_y:
            ax.set(xlabel="$y_1$", ylabel="$y_2$")
        else:
            ax.set(xlabel="$x$", ylabel="$y$")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    plot_path = os.path.join(PLOT_PATH,DATASET_NAME,"{}.png".format(file_name))
    plt.savefig(plot_path)
    plt.close(fig)

#create nn spec for discriminator and generator
config = {
    "noise_dim": 5,
    "epochs": 15000,
    "batch_size": 200,
    "gen_lr": 1e-4,
    "disc_lr": 1e-4,
    "val_interval": 100,
    "eval_batch_size": 1000,
    "eval_samples": 200,
    "kernel_scales": 50,
    "kernel_scale_min": 0.001,
    "kernel_scale_max": 0.5,
    "scatter": 0,
    "pdf_index":"3600",
    "kde_batch_size": 10,
    "n_critic": 5,
    "lambda_gp": 0.05
}
nn_spec = {'gen_spec' : {
    "other_dim": config["noise_dim"],#noise dimensions
    "cond_dim": X_DIM,#conditioning data
    "nodes_per_layer": [40,40,40,40],
    "output_dim": Y_DIM,#fake data dimensions
    "activation": nn.ReLU(),
    "type": FeedForward,
    "dropout":"0",
    "activation_final": 0
},
'critic_spec': {
    "other_dim": Y_DIM,#actual data dimensions
    "cond_dim": X_DIM,
    "nodes_per_layer": [25,25,25,25],
    # "cond_layers": [64,64],
    # "other_layers":[64,64],
    "output_dim": 1,#output logit
    "activation":nn.ReLU(),
    "type": FeedForward,
    "dropout": '0',
    "activation_final": 0
}
}

print(config)
print(nn_spec)
savepath = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'learning_prog_idx{}'.format(config['pdf_index']))
if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    for f in os.listdir(savepath):
        os.remove(os.path.join(savepath,f))
wcgan_model = WCGAN(config, nn_spec)
wcgan_model.train(train_data, val_data, val_func)

# import raw data
path = os.path.join(DATASET_PATH,'raw_data/test/data_raw.dat')
df_test = pd.read_csv(path, header = 0, index_col = 0)
aero_test_raw = df_test.loc[:, ["URef", "PLExp", "IECturbc", CHANNEL_NAME]]
test_raw = LabelledData(x= aero_test_raw.to_numpy()[:,:X_DIM],y = aero_test_raw.to_numpy()[:,X_DIM:])

x_values_scale, x_values_index = np.unique(test_data.x, axis = 0, return_index=True)
x_values = np.unique(test_raw.x, axis = 0)
num_samples_gen = 3000
sort =np.argsort(x_values_index)
x_values_scale = x_values_scale[sort]
x_values_index = x_values_index[sort]
x_values = x_values[sort]

samplepdf_imgs_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'Sample_PDF')

if not os.path.exists(samplepdf_imgs_path):
    os.makedirs(samplepdf_imgs_path)
else:
    for f in os.listdir(samplepdf_imgs_path):
        os.remove(os.path.join(samplepdf_imgs_path,f))

assert os.path.exists(samplepdf_imgs_path),("dataset folder {} does not exist".format(samplepdf_imgs_path))

gen_samples = np.zeros((num_samples_gen,len(x_values_scale)))
real_samples = np.zeros((num_samples_real,len(x_values_scale)))

print('Plotting samples for all x-locations...')
for i, (idx,values_scaled) in enumerate(zip(x_values_index, x_values_scale)):
    gen_samples[:,i] = get_samples(wcgan_model, values_scaled, num_samples_gen).squeeze(1)
    plt.figure()
    sns.kdeplot(gen_samples[:,i], color ='b',label='Gen')
    tmp = indexes(test_data.x[idx], test_data.x)
    real_samples[:,i] = test_data.y[tmp].squeeze()
    sns.kdeplot(real_samples[:,i], color='k', linestyle='--', label='True')
    plt.title('x={}'.format(x_values[i]), fontsize=10)
    plt.tight_layout()
    plt.legend()
    plt.savefig('{}/idx_{}.png'.format(samplepdf_imgs_path, i))
    plt.close()
print('Plotting samples for all x-locations finished')

print('Writing samples...')
np.savetxt(os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'samples.csv') ,gen_samples, delimiter=',')
print('Done')
