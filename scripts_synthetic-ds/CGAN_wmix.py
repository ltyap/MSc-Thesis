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

if generate_new_data:
# Sampling from wmix
    x_samples = np.random.uniform(low=0.0, high=1.0, size=(n_samples, 6))
    combined = sample(x_samples)

    train_val_split = config['train_val_split']
    assert train_val_split < 1 and train_val_split > 0, 'incorrect "train_val_split" value given'    
    combined_train, combined_validation = model_selection.train_test_split(combined, test_size=train_val_split, random_state=43)

    scaler = StandardScaler()
    train_set_scaled = scaler.fit_transform(combined_train)
    # Scale test data based on training data
    val_set_scaled = scaler.transform(combined_validation)

    # Test set for W evaluation
    tmp = np.random.uniform(low=0.0, high=1.0, size=(config['num_test_points'], 6))
    x_samples_test_repeated = np.repeat(tmp, repeats=config['size_per_test_point'], axis = 0)
    combined_test = sample(x_samples_test_repeated)
    test_set_scaled = scaler.transform(combined_test)
    
    # Make sure that dataset from above corresponds to data imported below

    #export data
    DATASET_PATH = '{}/datasets'.format(parent)
    DATASET_NAME = 'wmix6'

    # preprocessing folder
    raw_dataset_path = os.path.join(DATASET_PATH,DATASET_NAME,'raw_data')

    if not os.path.exists(raw_dataset_path):
        os.makedirs(raw_dataset_path)

    # raw data
    np.savetxt("{}/{}/raw_data/train.csv".format(DATASET_PATH,DATASET_NAME), combined_train,delimiter=",")
    np.savetxt("{}/{}/raw_data/val.csv".format(DATASET_PATH,DATASET_NAME), combined_validation,delimiter=",")
    np.savetxt("{}/{}/raw_data/test.csv".format(DATASET_PATH,DATASET_NAME), combined_test,delimiter=",")

    # scaled (training) data
    np.savetxt("{}/{}/train.csv".format(DATASET_PATH,DATASET_NAME), train_set_scaled,delimiter=",")
    np.savetxt("{}/{}/val.csv".format(DATASET_PATH,DATASET_NAME), val_set_scaled,delimiter=",")
    np.savetxt("{}/{}/test.csv".format(DATASET_PATH,DATASET_NAME), test_set_scaled,delimiter=",")
    
    print('New data generated and saved')


#import data (DO NOT TOUCH)
DATASET_PATH = '{}/datasets'.format(parent)
DATASET_NAME = 'wmix6'
X_DIM = 6
Y_DIM = 1
header = ['x1','x2',"x3",'x4','x5','x6','y']

dataset_dir = os.path.join(DATASET_PATH,DATASET_NAME)
plot_directory = os.path.join("{}/plots".format(parent), DATASET_NAME)
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
# save_path = os.path.join("{}/plots".format(parent), DATASET_NAME)
save_path = plot_directory
list_of_splits = ["train","val","test"]
splits = import_data(dataset_dir, X_DIM, Y_DIM, list_of_splits, save_path= save_path)
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
PLT_DATASET_NAME = 'wmix6'

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