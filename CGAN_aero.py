import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from misc_funcs import indexes, get_samples
import dataset_specifications.dataset as Dataset
import dataset_list
from networks import NoiseInjection, FeedForward, DoubleInputNetwork
from cgan_versions import PearsonCGAN, KLCGAN, RKLCGAN, WCGAN, WdivCGAN
import seaborn as sns
import pandas as pd
import evaluation as eval


plt.rcParams.update({'font.size': 13})
plt.rc("axes", labelsize=18)
plt.rc("axes", titlesize=21)
plt.rc("legend", fontsize=18)

#import data
name = 'aero'
DATASET_PATH = './datasets/{}'.format(name)
assert os.path.exists(DATASET_PATH),("dataset folder {} does not exist".format(DATASET_PATH))

print("Dataset path:", DATASET_PATH)
dataset = dataset_list.get_dataset_spec(name)()

"""
'TwrBsMyt_[kN-m] mean', 'TwrBsMyt_[kN-m] max', 'TwrBsMyt_[kN-m] stddev','TwrBsMyt_[kN-m] ST_DEL', 
'RootMyb1_[kN-m] mean', 'RootMyb1_[kN-m] max', 'RootMyb1_[kN-m] stddev','RootMyb1_[kN-m] ST_DEL',
'RootMxb1_[kN-m] mean', 'RootMxb1_[kN-m] max', 'RootMxb1_[kN-m] stddev', 'RootMxb1_[kN-m] ST_DEL',
'YawBrMyn_[kN-m] mean', 'YawBrMyn_[kN-m] max', 'YawBrMyn_[kN-m] stddev', 'YawBrMyn_[kN-m] ST_DEL'
"""

list_of_channels = dataset.channels
CHANNEL_NAME = list_of_channels[11]
print("Channel name:", CHANNEL_NAME)
DATASET_NAME = dataset.key[CHANNEL_NAME]

runs = 1
folder_names = ['WCGAN-A3']
models = [WCGAN]
for (folder, model) in zip(folder_names, models): 
    for run in range(runs):
        print("Run no.:", run)

        # For saving plots
        PLOT_PATH = './plots'
        PLT_DATASET_NAME = '{}/{}'.format(name,DATASET_NAME)

        # path for saving parameters of model
        PARAM_PATH = './param_best/{}/{}'.format(name, DATASET_NAME)
        FILE_NAME = '{}/RUN-{}'.format(folder, run)#run

        #CHANGE DIMENSIONS OF DATA ACCORDINGLY
        X_DIM = dataset.x_dim
        Y_DIM = dataset.y_dim

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
        dataset_dir = constants['dataset_path']
        assert os.path.exists(dataset_dir),("dataset folder {} does not exist".format(dataset_dir))

        splits = {}
        scatter_plot = 0
        for split in ("train","test","val"):
            data_path = os.path.join(dataset_dir,"{}.csv".format(split))
            assert os.path.exists(data_path),"data file {} does not exist".format(data_path)

            data = pd.read_csv(data_path,delimiter=",")
            data = data[dataset.inputs+[CHANNEL_NAME]].to_numpy()
            # For 1D data only
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
            return eval.evaluate_model(model, data_val = val_data, data_train = train_data, data_test = test_data, epoch = epoch)

        config = {
            "noise_dim": 30,
            "noise_dist": 'gaussian',
            "epochs": 30000,
            "batch_size": 200,
            "gen_lr": 2e-4,
            "disc_lr": 1e-4,
            "val_interval": 100,
            "eval_batch_size": 1000,    
            "eval_samples": 200,
            "kernel_scales": 50,
            "kernel_scale_min": 0.001,
            "kernel_scale_max": 0.7,
            "pdf_index":"100",
            "scatter": 0,
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
            "batch_norm": None,
            "spectral_normalisation": None
        },
        'disc_spec': {
            "other_dim": Y_DIM,#actual data dimensions
            "cond_dim": X_DIM,    
            "nodes_per_layer": [64,64,64,64,64,64,64],
            # "cond_layers": [64,64],
            # "other_layers":[64,64],
            "output_dim": 1,#output logit
            "activation":nn.ReLU(),
            "type": FeedForward,
            "dropout": None,
            "activation_final": 0,
            "batch_norm": None,
            "spectral_normalisation": None
        }
        }

        savepath = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'learning_prog_idx{}'.format(config['pdf_index']))
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        else:
            for f in os.listdir(savepath):
                os.remove(os.path.join(savepath,f))
        print(config)
        print(nn_spec)
        cgan_model = model(config, nn_spec, constants)
        print('CGAN model:', cgan_model)
        cgan_model.train(train_data, val_data, test_data, val_func)

        # import raw data
        path = 'datasets/{}/raw_data/test/data_raw.dat'.format(name)
        assert os.path.exists(path),("raw dataset folder {} does not exist".format(dataset_dir))

        df_test = pd.read_csv(path, header = 0, index_col = 0)
        aero_test_raw = df_test.loc[:, ["URef", "PLExp", "IECturbc", CHANNEL_NAME]]
        test_raw = Dataset.LabelledData(x= aero_test_raw.to_numpy()[:,:X_DIM],y = aero_test_raw.to_numpy()[:,X_DIM:])

        x_values_scale, x_values_index, counts = np.unique(test_data.x, axis = 0, return_counts = True, return_index=True)
        x_values = np.unique(test_raw.x, axis = 0)
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
        tmp = pd.read_csv("datasets/{}/raw_data/train/data_raw.dat".format(dataset.name),header=0, index_col=0)
        scaling = tmp.loc[:, CHANNEL_NAME]
        scaling_mean = scaling.mean()
        scaling_std = scaling.std()
        rescaled_samples = gen_samples*scaling_std+scaling_mean
        rescaled_samples_ = rescaled_samples.reshape(-1,1, order='F')
        combined = np.hstack((x_values_repeated,rescaled_samples_))
        df_combined = pd.DataFrame(combined, columns = dataset.inputs+[CHANNEL_NAME])

        print('Writing samples...')
        path = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME)
        df_combined.to_csv(os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,'samples.csv'))
        print('Done')