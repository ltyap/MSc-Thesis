import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from misc_funcs import indexes, get_samples
import dataset_specifications.dataset as Dataset
from networks import NoiseInjection, FeedForward, DoubleInputNetwork
from cgan_versions import PearsonCGAN
import seaborn as sns
import pandas as pd
import evaluation as eval

plt.rcParams.update({'font.size': 13})
plt.rc("axes", labelsize=18)
plt.rc("axes", titlesize=21)
plt.rc("legend", fontsize=18)

#import data
DATASET_PATH = './datasets/aerohydro/'
DATASET_NAME = 'YawBrMyn_ST_DEL'
CHANNEL_NAME = "YawBrMyn_[kN-m] ST_DEL"

# For saving plots
PLOT_PATH = './plots'
PLT_DATASET_NAME = 'aerohydro_{}'.format(DATASET_NAME)

# path for saving parameters of model
PARAM_PATH = './param_best'
FILE_NAME = 'aerohydro_pearsoncgan_nd_30_modified'

#CHANGE DIMENSIONS OF DATA ACCORDINGLY
X_DIM = 6
Y_DIM = 1

config = {
    "noise_dim": 30,
    "epochs": 1000,
    "batch_size": 400,
    "gen_lr": 1e-4,
    "disc_lr": 1e-4,
    "val_interval": 50,
    "eval_batch_size": 1000,
    "eval_samples": 200,
    "kernel_scales": 50,
    "kernel_scale_min": 0.001,
    "kernel_scale_max": 0.5,
    "scatter": 0,
    "pdf_index":"2400",
    "kde_batch_size": 10,
    # "n_critic": 20,
    # "lambda_gp": 5e-2,
    # "one-sided":False
}
nn_spec = {'gen_spec' : {
    "other_dim": config["noise_dim"],#noise dimensions
    "cond_dim": X_DIM,#conditioning data
    "nodes_per_layer": [128, 128, 128, 128, 128, 128, 128, 128],
    "output_dim": Y_DIM,#fake data dimensions
    "activation": nn.ReLU(),
    "type": NoiseInjection,
    "dropout":None,
    "activation_final": 0,
    "spectral_normalisation": None,

},
'disc_spec': {
    "other_dim": Y_DIM,#actual data dimensions
    "cond_dim": X_DIM,
    "nodes_per_layer": [128, 128, 64, 64, 64, 64],
    "cond_layers": [64,64],
    "other_layers":[64,64],
    "output_dim": 1,#output logit
    "activation":nn.ReLU(),
    "type": DoubleInputNetwork,
    "dropout": None,
    "activation_final": 0,
    "spectral_normalisation": True,
}
}

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
num_samples_real = 300
dataset_dir = os.path.join(constants['dataset_path'],constants['dataset_name'])
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
    splits[split] = Dataset.LabelledData(x=torch_data[:,:X_DIM],y=torch_data[:,X_DIM:])

train_data = splits["train"]
val_data = splits['val']
test_data = splits['test']

def val_func(model, epoch):
    return eval.evaluate_model(model, data = val_data, data_train = train_data, data_test = test_data, epoch = epoch)



print(config)
print(nn_spec)
savepath = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'learning_prog_idx{}'.format(config['pdf_index']))
if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    for f in os.listdir(savepath):
        os.remove(os.path.join(savepath,f))
model = PearsonCGAN(config, nn_spec, constants)
model.train(train_data, val_func)

# import raw data
path = os.path.join(DATASET_PATH,'raw_data/test/data_raw.dat')
df_test = pd.read_csv(path, header = 0, index_col = 0)
aero_test_raw = df_test.loc[:, ["URef", "PLExp", "IECturbc","WaveHs","WaveTp","WaveDir", CHANNEL_NAME]]
test_raw = Dataset.LabelledData(x= aero_test_raw.to_numpy()[:,:X_DIM],y = aero_test_raw.to_numpy()[:,X_DIM:])

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
    gen_samples[:,i] = get_samples(model, values_scaled, num_samples_gen).squeeze(1)
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
