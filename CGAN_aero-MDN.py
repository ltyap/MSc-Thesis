import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from misc_funcs import import_data, visualisation, get_config, learning_cleanup
import dataset_specifications.dataset as Dataset
import dataset_list
from networks import NoiseInjection, FeedForward, DoubleInputNetwork
from cgan_versions import PearsonCGAN, KLCGAN, RKLCGAN, WCGAN, CGAN
import pandas as pd
import evaluation as eval

plt.rcParams.update({'font.size': 13})
plt.rc("axes", labelsize=18)
plt.rc("axes", titlesize=21)
plt.rc("legend", fontsize=18)

config, nn_spec = get_config()
############################################################
# TO MODIFY IF NEEDED
"""
Details of this dataset, including the preprocessing of the raw data are in './dataset_specifications/aero_MDN.py'.
Only two things need to be changed in this script: the channel index (channel_index) and the training set (selection).
The list of channels available for training are as given in './dataset_specifications/aero_MDN.py':

['TwrBsMyt_[kN-m] max', 'TwrBsMyt_[kN-m] stddev','TwrBsMyt_[kN-m] ST_DEL', 
'RootMyb1_[kN-m] mean', 'RootMyb1_[kN-m] max', 'RootMyb1_[kN-m] stddev','RootMyb1_[kN-m] ST_DEL',
'RootMxb1_[kN-m] mean', 'RootMxb1_[kN-m] max', 'RootMxb1_[kN-m] stddev', 'RootMxb1_[kN-m] ST_DEL',
'YawBrMyn_[kN-m] mean', 'YawBrMyn_[kN-m] max', 'YawBrMyn_[kN-m] stddev', 'YawBrMyn_[kN-m] ST_DEL']

The indexes of each load channel are based on the order they appear in the list above. 
TwrBsMyt max is index 0, YawBrMyn ST_DEL is the last index (-1)
"""
channel_index = 2 

'''
The training sizes available are: 685, 1100, 2200, 5700, 11000, 23000, 45000.
More training sizes can be added, as long as they follow the same directory structure.
Multiple training sizes can be entered; the script will loop over each training set.
'''
selections = ['5700']

name = 'aero_MDN'#---> to change depending on the dataset; requires relevant dataset specification file

dataset = dataset_list.get_dataset_spec(name)()
list_of_channels = dataset.channels
CHANNEL_NAME = list_of_channels[channel_index]
print("Channel name:", CHANNEL_NAME)
DATASET_NAME = dataset.folder_names[CHANNEL_NAME]
DATASET_INPUTS = dataset.inputs 
selection_of_cols = DATASET_INPUTS+[CHANNEL_NAME]
X_DIM = dataset.x_dim
Y_DIM = dataset.y_dim

##########################################################################################
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
nn_spec['gen_spec']['type']= available_architectures[nn_spec['gen_spec']['type']]

nn_spec['disc_spec']['type']= available_architectures[nn_spec['disc_spec']['type']]
nn_spec['disc_spec']['activation']= available_activations[nn_spec['disc_spec']['activation']]

nn_spec['gen_spec']['other_dim'] = config["noise_dim"] #dimensions of noise
nn_spec['gen_spec']['cond_dim'] = X_DIM #dimensions of conditional data
nn_spec['gen_spec']['output_dim'] = Y_DIM #dimensions of generated data

nn_spec['disc_spec']['other_dim'] = Y_DIM#dimensions of actual data
nn_spec['disc_spec']['cond_dim'] = X_DIM#dimensions of conditional data
nn_spec['disc_spec']['output_dim'] = 1 

runs = config['runs'] # Number of runs per model

# import raw data
path = 'datasets/{}/raw_data/test/data_raw.dat'.format(name)
assert os.path.exists(path),("raw dataset folder {} does not exist".format(path))

df_test = pd.read_csv(path, header = 0, index_col = 0)
aero_test_raw = df_test.loc[:, selection_of_cols].to_numpy()
torch_data = torch.tensor(aero_test_raw, device='cpu').float()
test_raw = Dataset.LabelledData(x= torch_data[:,:X_DIM],y = torch_data[:,X_DIM:])

# Import actual Aero data for normalisation
df_train_raw = pd.read_csv("datasets/aero/raw_data/train/data_raw.dat",header=0, index_col=0)
aero_train_raw = df_train_raw.loc[:, selection_of_cols].to_numpy()
torch_data = torch.tensor(aero_train_raw, device='cpu').float()
train_raw = Dataset.LabelledData(x = torch_data[:,:X_DIM],y = torch_data[:,X_DIM:])


folder_names = config['folder_names']
models = []
for model in config['models']:
    models.append(available_models[model]) 
assert len(folder_names) == len(models), 'Number of models trained must equal number of folder names given'

for sel in selections:
    #import data
    selection = sel # number of training samples to use
    DATASET_PATH = './datasets/{}/{}'.format(name, selection)
    assert os.path.exists(DATASET_PATH),("dataset folder {} does not exist".format(DATASET_PATH))
    print('-------------------------------')
    print("Dataset path:", DATASET_PATH)
    list_of_splits = ["train","val","test"]
    splits = import_data(DATASET_PATH, X_DIM, Y_DIM, list_of_splits, selection = selection_of_cols)
    train_data = splits["train"]
    val_data = splits['val']
    test_data = splits['test']

    for (folder, model) in zip(folder_names, models):     
        for run in range(runs):
            print("Run no.:", run)
            # For saving plots
            PLOT_PATH = './plots'
            PLT_DATASET_NAME = '{}/{}'.format(name,DATASET_NAME)

            # path for saving parameters of model
            PARAM_PATH = './param_best/{}/{}'.format(name,DATASET_NAME)
            FILE_NAME = '{}/{}/RUN-{}'.format(folder,selection,run)

            constants = {
                "dataset_path": DATASET_PATH,
                "dataset_name": DATASET_NAME,
                "channel_name": CHANNEL_NAME,
                "plot_path": PLOT_PATH,
                "plt_dataset_name": PLT_DATASET_NAME,
                "param_path": PARAM_PATH,
                "file_name": FILE_NAME,
                "x_dim": X_DIM,
                "y_dim": Y_DIM
            }

            def val_func(model, epoch):
                return eval.evaluate_model(model, data_val = val_data, 
                                           data_train = train_data, data_test = test_data, epoch = epoch)
            
            # clean up before training
            learning_cleanup(constants, config)

            print(config)
            print(nn_spec)
            cgan_model = model(config, nn_spec, constants)
            print('CGAN model:', cgan_model)
            cgan_model.train(train_data, val_data, test_data, val_func)

            gen_samples_file = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'samples.csv')
            kde_vis_test_pts_path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'Sample_PDF')

            visualisation(test_data, train_data_raw = train_raw,
                           test_data_raw = test_raw, 
                          samplepdf_imgs_path = kde_vis_test_pts_path,
                   model = cgan_model, gen_samples_path = gen_samples_file, 
                   gen_samples_file_header = selection_of_cols,
                   num_samples_gen = 3000
                   )