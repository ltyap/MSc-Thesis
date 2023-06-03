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
# sns.set_theme(style="white")
# sns.set_context("talk")
import pandas as pd
import evaluation as eval
import scipy.stats as ss
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.size': 13})
plt.rc("axes", labelsize=18)
plt.rc("axes", titlesize=21)
plt.rc("legend", fontsize=18)

# Gaussian mixtures
def mu1(x):
    return x+1 
def mu2(x):
    # return 2*x+1
    return 2*x+2.5
def mu3(x):
    return 4*x+5
def mu4(x):
    return 1.8*x+10
def std1(x): 
    return 0.2
def std2(x):
    # return np.sqrt(x+1.5)
    return 0.2+0.3*x
def std3(x):
    return 0.1+0.7*x
def std4(x):
    return 0.8-0.5*x
def pdf_func(x, x_supp, mixture_params, mixture_weights):
    y_pdf = np.zeros_like(x_supp)
    for (l, s), w in zip(mixture_params, mixture_weights):
        y_pdf += ss.norm.pdf(x_supp, loc=l(x), scale=s(x))*w
    return y_pdf

def sample_ys(x_samples,
            mixture_params,
            mixture_idx,
            n):
    assert len(x_samples) == n, 'Sizes are not the same'
    # Sampling from GM
    y_samples = []
    for idx, i in enumerate(mixture_idx):
        mean = mixture_params[i,0]
        std = mixture_params[i,1]
        sample = ss.norm.rvs(loc=mean(x_samples[idx]), scale=std(x_samples[idx]))
        y_samples.append(sample)
    return np.array(y_samples)

def sample(xs = np.random.uniform(low=0, high=1, size=5000),
           mixture_weights = np.array([0.1,0.4,0.2,0.3]),
           mixture_params = np.array([[mu1,std1],[mu2, std2],[mu3,std3],[mu4,std4]]),
           ):

    mixture_n_comp = mixture_params.shape[0]
    n = len(xs)
    assert mixture_n_comp == len(mixture_weights), 'Not the same size'
    mixture_idx = np.random.choice(mixture_n_comp, size = n, p = mixture_weights)

    ys = sample_ys(xs, mixture_params, mixture_idx, n)
    # print(ys.shape)
    # print(xs.shape)
    return np.stack((xs, ys), axis=1)

params = np.array([[mu1,std1],[mu2, std2],[mu3,std3],[mu4,std4]])#
weights = np.array([0.1,0.4,0.2,0.3])

generate = False
if generate:
    # Sampling from 1D-GM 
    n = 2000 #number of samples
    x_samples = np.random.uniform(low=0, high=1, size=n)
    combined = sample(x_samples, mixture_weights= weights, mixture_params = params)

    combined_train, combined_validation = model_selection.train_test_split(combined, test_size=0.5, random_state=43)

    scaler = StandardScaler()
    train_set_scaled = scaler.fit_transform(combined_train)
    # Scale test data based on training data
    val_set_scaled = scaler.transform(combined_validation)


    # Test set for W evaluation
    tmp = np.random.uniform(low=0, high=1, size=50)
    tmp = np.sort(tmp, axis=0)
    x_samples_test_repeated = np.repeat(tmp, repeats=200, axis = 0)
    x_samples_test_repeated = x_samples_test_repeated.reshape((len(x_samples_test_repeated,)))
    combined_test = sample(x_samples_test_repeated)
    test_set_scaled = scaler.transform(combined_test)

    # Make sure that heteroskedastic set from above corresponds to data imported below

    #export data
    DATASET_PATH = './datasets'
    DATASET_NAME = '1D-GM'

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
DATASET_NAME = '1D-GM'
dataset_dir = os.path.join(DATASET_PATH,DATASET_NAME)
assert os.path.exists(dataset_dir),("dataset folder {} does not exist".format(dataset_dir))

splits = {}
#CHANGE DIMENSIONS OF DATA ACCORDINGLY
X_DIM = 1
Y_DIM = 1
scatter_plot = 1

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

raw_dataset_folder = os.path.join('./datasets','1D-GM','raw_data')

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
for run in range(runs):
    print("Run no.:", run)

    # path for saving parameters of model
    PARAM_PATH = './param_best/1D-GM'
    FILE_NAME = 'WCGAN-A3/RUN-{}'.format(run)

    # For saving plots
    PLOT_PATH = './plots'
    PLT_DATASET_NAME = '1D-GM'

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
        "noise_dim": 5,
        "noise_dist": 'gaussian',
        "epochs": 3000,
        "batch_size": 200,
        "gen_lr": 2e-4,
        "disc_lr": 1e-4,
        "val_interval": 20,
        "eval_batch_size": 1000,
        "eval_samples": 200,
        "kernel_scales": 50,
        "kernel_scale_min": 0.001,
        "kernel_scale_max": 0.7,
        "pdf_index":0,
        "scatter": 1,
        "kde_batch_size": 10,    
        "n_critic": 5,
        "lambda_gp": 2e-2,
        'one-sided': True
    }

    nn_spec = {'gen_spec' : {
        "other_dim": config["noise_dim"],#noise dimensions
        "cond_dim": X_DIM,#conditioning data
        "nodes_per_layer": [64,64,64,64,64,64,64],
        "output_dim": Y_DIM,#fake data dimensions
        "activation": nn.ReLU(),
        "type": FeedForward,
        "dropout":None,
        "activation_final": 0,
        "spectral_normalisation": None,
        "batch_norm": None
    },
    'disc_spec': {
        "other_dim": Y_DIM,#actual data dimensions
        "cond_dim": X_DIM,
        "nodes_per_layer": [64,64,64,64,64,64,64],
        # "cond_layers": [64,64],
        # "other_layers":[64,64],
        "output_dim": 1,#output logit
        "activation": nn.ReLU(),
        "type": FeedForward,
        "dropout":None,
        "activation_final": 0,
        "spectral_normalisation": None,
        "batch_norm": None
    }
    }
    print(config)
    print(nn_spec)
    cgan_model = WCGAN(config, nn_spec, constants)
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
        sns.kdeplot(gen_samples[:,i], color ='b',label='Gen', bw_adjust=0.4)
        sns.kdeplot(test_data.y[start_idx[i]:end_idx[i]].squeeze(), color='k', linestyle='--', label='True', bw_adjust=0.4)
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
    df_combined = pd.DataFrame(combined, columns = ['x','y'])

    print('Writing samples...')
    path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME)
    df_combined.to_csv(os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'samples.csv'))
    print('Done')