import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from misc_funcs import get_samples
from dataset_specifications.dataset import LabelledData
from networks import NoiseInjection, FeedForward, DoubleInputNetwork 
from cgan_versions import PearsonCGAN, KLCGAN, RKLCGAN, WCGAN, WdivCGAN
from cgan import CGAN
import seaborn as sns
import pandas as pd
import evaluation as eval
import scipy.stats as ss
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.size': 13})
plt.rc("axes", labelsize=18)
plt.rc("axes", titlesize=21)
plt.rc("legend", fontsize=18)

# wmix6
def sample_ys(xs):
    comp_offsets= xs[:,:2]
    weights = xs[:,2:4]
    comp_shape_mods = xs[:,4:]

    # Component probabilities (normalized weights)
    comp_probs = weights / np.sum(weights, axis=1, keepdims=True)
    comp_i = np.apply_along_axis(
        (lambda probs: np.random.choice(2, size=1, p=probs)),
            1, comp_probs) # Axis should be 1, see numpy docs
    # Shape (n, 1)

    # Get values for chosen component (component with sampled index)
    offsets = np.take_along_axis(comp_offsets, comp_i, 1)
    shape_mods = np.take_along_axis(comp_shape_mods, comp_i, 1)
    shape_params = 1. + shape_mods

    w_samples = np.random.weibull(shape_params)
    # Shape (n, 1)
    ys = w_samples + 5.*offsets

    return ys

def sample(xs = np.random.uniform(low=0.0, high=1.0, size=(5000, 6))):
    n = len(xs)
    # xs is from Uniform(0,1)^(dim(x))
    assert n == len(xs), 'noise length should be same as xs length'    
    ys = sample_ys(xs)
    return np.concatenate((xs, ys), axis=1)

def get_pdf(x):
    # Perform pdf computation in pytorch
    if type(x) == torch.Tensor:
        x = x.to("cpu")
    else:
        x = torch.tensor(x)

    comp_offsets = x[:2]
    weights = x[2:4]
    comp_shape_mods = x[4:]

    comp_probs = weights / torch.sum(weights)

    shapes = 1.0 + comp_shape_mods
    scales = torch.ones_like(shapes)

    w_dists = torch.distributions.weibull.Weibull(scales, shapes, validate_args=False)

    def pdf(y):
        no_offsets = y - 5.0*comp_offsets
        positive = (no_offsets >= 0.0)
        # Only use probability density for positive samples (within support)
        # Pytorch gives positive density even outside support for some reason

        log_probs = w_dists.log_prob(no_offsets)

        filtered_probs = torch.exp(log_probs[positive])
        pd = torch.sum(filtered_probs * comp_probs[positive])
        return pd

    return pdf
def get_support():
    return (0.1,10.)

generate = False
num_samples_real = 300
if generate:
    # Sampling from wmix
    n = 6600 #number of samples
    x_samples = np.random.uniform(low=0.0, high=1.0, size=(n, 6))
    combined = sample(x_samples)
    combined_train, combined_validation = model_selection.train_test_split(combined, test_size=0.5, random_state=43)

    scaler = StandardScaler()
    train_set_scaled = scaler.fit_transform(combined_train)
    # Scale test data based on training data
    val_set_scaled = scaler.transform(combined_validation)
    tmp = np.random.choice(len(combined_validation),50, replace=False)
    x_samples_test_repeated = np.repeat(combined_validation[tmp,:-1], repeats=num_samples_real, axis = 0)
    combined_test = sample(x_samples_test_repeated)
    test_set_scaled = scaler.transform(combined_test)
    
    # Make sure that dataset from above corresponds to data imported below

    #export data
    DATASET_PATH = './datasets'
    DATASET_NAME = 'wmix6'

    # preprocessing folder
    raw_dataset_path = os.path.join(DATASET_PATH,DATASET_NAME,'raw_data')

    if not os.path.exists(raw_dataset_path):
        os.makedirs(raw_dataset_path)

    # raw data
    np.savetxt("./{}/{}/raw_data/train.csv".format(DATASET_PATH,DATASET_NAME), combined_train,delimiter=",")
    np.savetxt("./{}/{}/raw_data/val.csv".format(DATASET_PATH,DATASET_NAME), combined_validation,delimiter=",")
    np.savetxt("./{}/{}/raw_data/test.csv".format(DATASET_PATH,DATASET_NAME), combined_test,delimiter=",")


    np.savetxt("./{}/{}/train.csv".format(DATASET_PATH,DATASET_NAME), train_set_scaled,delimiter=",")
    np.savetxt("./{}/{}/val.csv".format(DATASET_PATH,DATASET_NAME), val_set_scaled,delimiter=",")
    np.savetxt("./{}/{}/test.csv".format(DATASET_PATH,DATASET_NAME), test_set_scaled,delimiter=",")

#import data
DATASET_PATH = './datasets'
DATASET_NAME = 'wmix6'
dataset_dir = os.path.join(DATASET_PATH,DATASET_NAME)
assert os.path.exists(dataset_dir),("dataset folder {} does not exist".format(dataset_dir))

splits = {}
#CHANGE DIMENSIONS OF DATA ACCORDINGLY
X_DIM = 6
Y_DIM = 1
scatter_plot = 0

for split in ("train","val","test"):
    data_path = os.path.join(dataset_dir,"{}.csv".format(split))
    assert os.path.exists(data_path),"data file {} does not exist".format(data_path)
    
    data = np.genfromtxt(data_path,delimiter=",")
    if scatter_plot:
        plt.figure
        plt.scatter(data[:,:1],data[:,1:], c='k', s=3)
        plt.xlabel("x")
        plt.ylabel('y')
        plt.title(split)
        save_path = os.path.join("./plots", DATASET_NAME)
        plt.savefig('{}/{}.png'.format(save_path,split))
        plt.close()
    torch_data = torch.tensor(data, device="cpu").float()
    splits[split] = LabelledData(x=torch_data[:,:X_DIM],y=torch_data[:,X_DIM:])

train_data = splits["train"]
val_data = splits['val']
test_data = splits['test']

def val_func(model, epoch):
    return eval.evaluate_model(model, data_val = val_data, data_train = train_data,
                                data_test = test_data, epoch = epoch)

raw_dataset_folder = os.path.join('./datasets','wmix6','raw_data')

for split in ('train','test'):
    raw_data_path = os.path.join(raw_dataset_folder,"{}.csv".format(split))
    assert os.path.exists(raw_data_path),"raw data file {} does not exist".format(data_path)
    data = np.genfromtxt(raw_data_path,delimiter=",")
    torch_data = torch.tensor(data, device="cpu").float()
    splits[split] = LabelledData(x=torch_data[:,:X_DIM],y=torch_data[:,X_DIM:])

train_data_raw = splits["train"]
test_data_raw = splits["test"]
# For rescaling back to original scale
y_train_mean = torch.mean(train_data_raw.y).item()
y_train_std = torch.std(train_data_raw.y).item()

x_train_std = torch.std(train_data_raw.x).item()
x_train_mean = torch.mean(train_data_raw.x).item()

runs = 10
models = [
    CGAN,
    KLCGAN,
    PearsonCGAN,
    KLCGAN
    ]
model_file_name = ['CGAN', 'KLCGAN', 'PearsonCGAN', "RKLCGAN"]

for (name ,model) in zip(model_file_name, models):
    for run in range(runs):
        print("Run no.:", run)

        # path for saving parameters of model
        PARAM_PATH = './param_best/wmix6'
        FILE_NAME = '{}/RUN-{}'.format(name,run)

        # For saving plots
        PLOT_PATH = './plots'
        PLT_DATASET_NAME = 'wmix6'

        constants = {
        "dataset_path": DATASET_PATH,
        "dataset_name": DATASET_NAME,
        "plot_path": PLOT_PATH,
        "plt_dataset_name": PLT_DATASET_NAME,
        "param_path": PARAM_PATH,
        "file_name": FILE_NAME,
        "x_dim": X_DIM,
        "y_dim": Y_DIM
        }

        # #create nn spec for discriminator and generator
        config = {
            "noise_dim": 30,
            "noise_dist": 'gaussian',
            "epochs": 2000,
            "batch_size": 200,
            "gen_lr": 1e-4,
            "disc_lr": 1e-4,
            "val_interval": 20,
            "eval_batch_size": 1000,
            "eval_samples": 200,
            "kernel_scales": 50,
            "kernel_scale_min": 0.001,
            "kernel_scale_max": 0.7,
            "pdf_index":'100',
            "scatter": 0,
            "kde_batch_size": 10,    
            "n_critic": 5,
            "lambda_gp": 2e-2,
            'one-sided': True
        }

        nn_spec = {'gen_spec' : {
            "other_dim": config["noise_dim"],#noise dimensions
            "cond_dim": X_DIM,#conditioning data
            "nodes_per_layer": [64,64,64,64,64,64],
            "output_dim": Y_DIM,#fake data dimensions
            "activation": nn.ReLU(),
            "type": NoiseInjection,
            "dropout":None,
            "activation_final": 0,
            "spectral_normalisation": None,
            "batch_norm": None
        },
        'disc_spec': {
            "other_dim": Y_DIM,#actual data dimensions
            "cond_dim": X_DIM,
            "nodes_per_layer": [64,64,64,64,64,64],
            "cond_layers": [64,64],
            "other_layers":[64,64],
            "output_dim": 1,#output logit
            "activation": nn.ReLU(),
            "type": DoubleInputNetwork,
            "dropout":None,
            "activation_final": 0,
            "spectral_normalisation": None,
            "batch_norm": None
        }
        }
        print(config)
        print(nn_spec)
        cgan_model = model(config, nn_spec, constants)
        print('cgan model:',cgan_model)
        cgan_model.train(train_data, val_data, test_data, val_func)

        x_values_scale, x_values_index, counts = np.unique(test_data.x, axis = 0, return_counts = True, return_index=True)
        x_values = np.unique(test_data_raw.x, axis = 0)
        num_samples_gen = 3000
        sort =np.argsort(x_values_index)
        x_values_scale = x_values_scale[sort]
        x_values_index = x_values_index[sort]
        x_values = x_values[sort]
        start_idx = x_values_index
        end_idx = counts+x_values_index
        samplepdf_imgs_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'Sample_PDF')

        if not os.path.exists(samplepdf_imgs_path):
            os.makedirs(samplepdf_imgs_path)
        else:
            for f in os.listdir(samplepdf_imgs_path):
                os.remove(os.path.join(samplepdf_imgs_path,f))

        assert os.path.exists(samplepdf_imgs_path),("results folder {} does not exist".format(samplepdf_imgs_path))

        gen_samples = np.zeros((num_samples_gen,len(x_values_scale)))

        print('Plotting samples for all x-locations...')
        for i, (idx,values_scaled) in enumerate(zip(x_values_index, x_values_scale)):
            gen_samples[:,i] = get_samples(cgan_model, values_scaled, num_samples_gen).squeeze(1)
            plt.figure()
            sns.kdeplot(gen_samples[:,i], color ='b',label='Gen')
            sns.kdeplot(test_data.y[start_idx[i]:end_idx[i]].squeeze(), color='k', linestyle='--', label='True')
            plt.title('x={}'.format(x_values[i]), fontsize=10)
            plt.tight_layout()
            plt.legend()
            plt.savefig('{}/idx_{}.png'.format(samplepdf_imgs_path, i))
            plt.close()
        print('Plotting samples for all x-locations finished')

        x_values_repeated = x_values.repeat(num_samples_gen, axis=0)
        rescaled_samples = gen_samples*y_train_std+y_train_mean
        rescaled_samples_ = rescaled_samples.reshape(-1,1, order='F')
        combined = np.hstack((x_values_repeated,rescaled_samples_))
        df_combined = pd.DataFrame(combined)

        print('Writing samples...')
        path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME)
        df_combined.to_csv(os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'samples.csv'))
        print('Done')