import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as ss
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

path = os.getcwd()
# module directory
parent = os.path.dirname(path)
sys.path.append(parent)
from misc_funcs import get_samples, import_data, visualisation, get_config, learning_cleanup
from dataset_specifications.dataset import LabelledData
from networks import NoiseInjection, FeedForward, DoubleInputNetwork 
from cgan_versions import PearsonCGAN, KLCGAN, RKLCGAN, WCGAN
from cgan import CGAN
import evaluation as eval


plt.rcParams.update({'font.size': 13})
plt.rc("axes", labelsize=18)
plt.rc("axes", titlesize=21)
plt.rc("legend", fontsize=18)

config, nn_spec = get_config()

##########################################################################################
# # For generating new data
generate_new_data = config['new_data'] # Flag
n_samples = config['new_data_size'] # size of new dataset
###########################################################################################


available_models = {
    "cgan": CGAN,
    "klcgan": KLCGAN,
    "rklcgan": RKLCGAN,
    "pearson": PearsonCGAN,
    "wcgan": WCGAN
}
available_architectures = {
    "feedforward": FeedForward,
    "double-input": DoubleInputNetwork,
    "noise-injection": NoiseInjection
    }
available_activations ={
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    'elu': nn.ELU()    
}
nn_spec['gen_spec']['activation']= available_activations[nn_spec['gen_spec']['activation']]
nn_spec['disc_spec']['activation']= available_activations[nn_spec['disc_spec']['activation']]

nn_spec['gen_spec']['type']= available_architectures[nn_spec['gen_spec']['type']]
nn_spec['disc_spec']['type']= available_architectures[nn_spec['disc_spec']['type']]

folder_names = config['folder_names']
models = []
for model in config['models']:
    models.append(available_models[model]) 
assert len(folder_names) == len(models), 'Number of models trained must equal number of folder names given'
runs = config['runs'] # Number of runs per model

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


if generate_new_data:
    params = np.array([[mu1,std1],[mu2, std2],[mu3,std3],[mu4,std4]])#
    weights = np.array([0.1,0.4,0.2,0.3])
    # Sampling from 1D-GM   
    x_samples = np.random.uniform(low=0, high=1, size=n_samples)
    combined = sample(x_samples, mixture_weights= weights, mixture_params = params)

    train_val_split = config['train_val_split']
    assert train_val_split < 1 and train_val_split > 0, 'incorrect "train_val_split" value given'

    combined_train, combined_validation = model_selection.train_test_split(combined, test_size=train_val_split, random_state=43)

    scaler = StandardScaler()
    train_set_scaled = scaler.fit_transform(combined_train)
    # Scale test data based on training data
    val_set_scaled = scaler.transform(combined_validation)


    # Test set for W evaluation
    tmp = np.random.uniform(low=0, high=1, size=config['num_test_points'])
    tmp = np.sort(tmp, axis=0)
    x_samples_test_repeated = np.repeat(tmp, repeats=config['size_per_test_point'], axis = 0)
    x_samples_test_repeated = x_samples_test_repeated.reshape((len(x_samples_test_repeated,)))
    combined_test = sample(x_samples_test_repeated)
    test_set_scaled = scaler.transform(combined_test)

    #export data
    DATASET_PATH = '{}/datasets'.format(parent)
    DATASET_NAME = '1D-GM'

    # preprocessing folder
    raw_dataset_path = os.path.join(DATASET_PATH,DATASET_NAME,'raw_data')

    if not os.path.exists(raw_dataset_path):
        os.makedirs(raw_dataset_path)

    # raw data
    np.savetxt("{}/{}/raw_data/train.csv".format(DATASET_PATH,DATASET_NAME), combined_train,delimiter=",")
    np.savetxt("{}/{}/raw_data/val.csv".format(DATASET_PATH,DATASET_NAME), combined_validation,delimiter=",")
    np.savetxt("{}/{}/raw_data/test.csv".format(DATASET_PATH,DATASET_NAME), combined_test,delimiter=",")


    np.savetxt("{}/{}/train.csv".format(DATASET_PATH,DATASET_NAME), train_set_scaled,delimiter=",")
    np.savetxt("{}/{}/val.csv".format(DATASET_PATH,DATASET_NAME), val_set_scaled,delimiter=",")
    np.savetxt("{}/{}/test.csv".format(DATASET_PATH,DATASET_NAME), test_set_scaled,delimiter=",")

#import data (DO NOT TOUCH)
DATASET_PATH = '{}/datasets'.format(parent)
DATASET_NAME = '1D-GM'
X_DIM = 1
Y_DIM = 1
header = ['x','y']

dataset_dir = os.path.join(DATASET_PATH,DATASET_NAME)
plot_directory = os.path.join("{}/plots".format(parent), DATASET_NAME)
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
# save_path = os.path.join("{}/plots".format(parent), DATASET_NAME)
save_path = plot_directory
list_of_splits = ["train","val","test"]
splits = import_data(dataset_dir, X_DIM, Y_DIM, list_of_splits, scatter_plot = config['scatter'], save_path= save_path)
train_data = splits["train"]
val_data = splits['val']
test_data = splits['test']

# Load raw data
raw_dataset_dir = os.path.join(DATASET_PATH,DATASET_NAME,'raw_data')
list_of_splits = ["train","test"]
splits = import_data(raw_dataset_dir, X_DIM, Y_DIM, list_of_splits, training_data=False)
train_data_raw = splits["train"]
test_data_raw = splits["test"]

# Path for saving plots
PLOT_PATH = '{}/plots'.format(parent)
PLT_DATASET_NAME = '1D-GM'

constants = {
"dataset_path": DATASET_PATH,
"dataset_name": DATASET_NAME,
"plot_path": PLOT_PATH,
"plt_dataset_name": PLT_DATASET_NAME,
"x_dim": X_DIM,
"y_dim": Y_DIM
}

nn_spec['gen_spec']['other_dim'] = config["noise_dim"] #dimensions of noise
nn_spec['gen_spec']['cond_dim'] = X_DIM #dimensions of conditional data
nn_spec['gen_spec']['output_dim'] = Y_DIM #dimensions of generated data

nn_spec['disc_spec']['other_dim'] = Y_DIM#dimensions of actual data
nn_spec['disc_spec']['cond_dim'] = X_DIM#dimensions of conditional data
nn_spec['disc_spec']['output_dim'] = 1 

# # Validation function for training model
def val_func(model, epoch):
    return eval.evaluate_model(model, data_val = val_data, data_train = train_data,
                                data_test = test_data, epoch = epoch)

for (folder, model) in zip(folder_names, models):
    for run in range(runs):
        print("Run no.:", run)

        # path for saving parameters of model
        PARAM_PATH = '{}/param_best/{}'.format(parent,DATASET_NAME)
        FILE_NAME = '{}/RUN-{}'.format(folder, run)
        constants["param_path"] = PARAM_PATH
        constants["file_name"] = FILE_NAME

        print(config)
        print(nn_spec)
        learning_cleanup(constants, config)
        cgan_model = model(config, nn_spec, constants)
        print('CGAN model:', cgan_model)
        cgan_model.train(train_data, val_data, test_data, val_func)

        
        gen_samples_file = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'samples.csv')
        kde_vis_test_pts_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'Sample_PDF')

        visualisation(test_data, train_data_raw, test_data_raw, 
                      samplepdf_imgs_path = kde_vis_test_pts_path, 
                      model = cgan_model, gen_samples_path = gen_samples_file, 
                      gen_samples_file_header = header, num_samples_gen = config['gen_samples'])