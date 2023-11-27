import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import shutil
from dataset_specifications.dataset import LabelledData

# Get list of indexes associated with same data
def indexes(val, x):
    listofindexes = []
    for i in range(len(x)):
        if (x[i] == val).all():
            listofindexes.append(i)
    return listofindexes

#get samples from trained model
def get_samples(model,input,num_samples=1000):
    input = torch.tensor(input, dtype = torch.float)
    input_repeated = input.repeat(num_samples,1)
    samples = model.sample(input_repeated, batch_size = 1000)
    return samples.cpu()

def json_to_dict(path):
    with open(path) as json_file:
        json_dict = json.load(json_file)
    return json_dict

def read_spec(path):
    assert os.path.exists(path), (
            "Specification file '{}' does not exist".format(path))
    spec_dict = json_to_dict(path)

    return spec_dict


def import_data(dataset_dir, x_dim, y_dim, list_of_splits, scatter_plot = False, 
                save_path=None, training_data = True, selection=None):

    assert os.path.exists(dataset_dir),("dataset folder {} does not exist".format(dataset_dir))
    if training_data:
        print('Directory for loading training data:', dataset_dir)
    else:
        print('Directory for loading data:', dataset_dir)
    splits = {}

    if save_path != None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print('Save path for scatter plot of training data:', save_path)

    # Load data
    for split in list_of_splits:
        data_path = os.path.join(dataset_dir,"{}.csv".format(split))
        assert os.path.exists(data_path),"data file {} does not exist".format(data_path)
    
        # data = np.genfromtxt(data_path,delimiter=",")
        data = pd.read_csv(data_path,delimiter=",")
        if selection == None:
            data = data.to_numpy()
        else:
            data = data[selection].to_numpy()

        if scatter_plot:
            plt.figure
            plt.scatter(data[:,:x_dim],data[:,x_dim:], c='k', s=3)
            plt.xlabel("x")
            plt.ylabel('y')
            plt.title(split)
            plt.savefig('{}/{}.png'.format(save_path,split))
            plt.close()
        torch_data = torch.tensor(data, device="cpu").float()
        splits[split] = LabelledData(x=torch_data[:,:x_dim],y=torch_data[:,x_dim:])

    return splits

def visualisation(test_data, train_data_raw, test_data_raw, samplepdf_imgs_path,
                   model, gen_samples_path, 
                   gen_samples_file_header=None,
                   num_samples_gen = 3000,
                   bw = 1):
    # For rescaling training data back to original scale
    y_train_mean = torch.mean(train_data_raw.y).item()
    y_train_std = torch.std(train_data_raw.y).item()

    # x_train_std = torch.std(train_data_raw.x).item()
    # x_train_mean = torch.mean(train_data_raw.x).item()

    x_values_scale, x_values_index, counts = np.unique(test_data.x, axis = 0, return_counts = True, return_index=True)
    x_values = np.unique(test_data_raw.x, axis = 0)
    sort =np.argsort(x_values_index)
    x_values_scale = x_values_scale[sort]
    x_values_index = x_values_index[sort]
    x_values = x_values[sort]
    start_idx = x_values_index
    end_idx = counts+x_values_index

    if not os.path.exists(samplepdf_imgs_path):
        os.makedirs(samplepdf_imgs_path)
    else:
        for f in os.listdir(samplepdf_imgs_path):
            os.remove(os.path.join(samplepdf_imgs_path,f))

    assert os.path.exists(samplepdf_imgs_path),("results folder {} does not exist".format(samplepdf_imgs_path))

    gen_samples = np.zeros((num_samples_gen,len(x_values_scale)))

    print('Plotting samples for all x-locations...')
    for i, (idx,values_scaled) in enumerate(zip(x_values_index, x_values_scale)):
        gen_samples[:,i] = get_samples(model, values_scaled, num_samples_gen).squeeze(1)
        plt.figure()
        sns.kdeplot(gen_samples[:,i], color ='b',label='Gen', bw_adjust=bw)
        sns.kdeplot(test_data.y[start_idx[i]:end_idx[i]].squeeze(), color='k', linestyle='--', label='True', bw_adjust=bw)
        plt.title('x={}'.format(x_values[i]), fontsize=10)
        plt.tight_layout()
        plt.legend()
        plt.savefig('{}/idx_{}.png'.format(samplepdf_imgs_path, i))
        plt.close()
    print('Plotting samples for all test x-locations finished')

    x_values_repeated = x_values.repeat(num_samples_gen, axis=0)
    rescaled_samples = gen_samples*y_train_std+y_train_mean
    rescaled_samples_ = rescaled_samples.reshape(-1,1, order='F')
    combined = np.hstack((x_values_repeated,rescaled_samples_))
    if gen_samples_file_header == None:
        df_combined = pd.DataFrame(combined)
    else:
        df_combined = pd.DataFrame(combined, columns = gen_samples_file_header)

    print('Writing samples...')
    df_combined.to_csv(gen_samples_path)
    print('Done')
   
def get_config():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, help="Config file to read run config from")
    parser.add_argument("--runs", type=int, help='Number of runs to execute per model', default=1)
    # parser.add_argument("--dataset", type=str, help="Which dataset to use")
    parser.add_argument("--folder_names", type=str, help="Name(s) of folder to save results to")
    parser.add_argument("--models", type=str, help='cgan model(s) to be used; number of models specified should\
                        correspond to number of folder names given',\
                            choices=['wcgan','pearson','rklcgan','klcgan','cgan'])
    parser.add_argument("--architecture_dir", type=str, help="Path to .json file that contains the architecture of \
                        the generator and discriminator")
    parser.add_argument('--noise_dim', type=int, help='Noise dimensionality', default=5)
    parser.add_argument('--noise_dist', type=str, help='Noise distribution', default='gaussian',\
                         choices=['gaussian', 'exponential', 'uniform', 'lognormal','censorednormal'])
    parser.add_argument('--epochs', type=int, help='Training epochs', default=3000)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=200)
    parser.add_argument('--gen_lr', type=float, help='Generator learning rate', default=1e-4)
    parser.add_argument('--disc_lr', type=float, help='Discriminator learning rate', default=1e-4)
    parser.add_argument('--val_interval', type=int, help='Validation epoch interval', default=20)
    parser.add_argument('--sample_batch_size', type=int, default=1000,
                        help='Number of x''s to pass to generator at a single time')
    parser.add_argument('--eval_samples', type=int, default=200,
                        help="How many samples to draw for estimating KDE in evaluation")
    parser.add_argument('--kernel_scales', type=int, help="Number of kernel scales used for kde log-likelihood estimation",default=50)
    parser.add_argument('--kernel_scale_min', type=float, default=0.001)
    parser.add_argument('--kernel_scale_max', type=float, default=0.7)
    parser.add_argument('--pdf_index', type=str, help='List of indexes in test dataset for plotting training progress via conditional pdf')
    parser.add_argument('--scatter', type=bool, default=False, 
                        help='Flag for plotting scatter plot showing training progress for x-dim=1 data only')
    parser.add_argument('--pdf_index_prefix',type=str, default="cond_dist_learning_prog",
                        help='folder prefix for conditional pdf training progress')
    parser.add_argument('--scatter_prefix',type=str, default="scatter_learning_prog",
                        help="folder prefix for scatter plot training progress (x-dim = 1 data only)")
    parser.add_argument('--prog_prefix',type=str, default="PROGRESS",
                        help='folder prefix containing training progress plots')
    parser.add_argument('--kde_batch_size', type=int, default=10, help="How many kernels scales to compute KDE for at the same time")
    parser.add_argument('--gen_samples', type = int, default=3000,
                        help='Number of samples per test point to be generated after training. Also used for kde visualisation')

    # Synthetic dataset specific options
    parser.add_argument('--new_data', type=bool, default=False, help='Generate new data for synthetic datasets')
    parser.add_argument('--new_data_size', type=int, default=2000, help='Number of new data to be generated for synthetic datasets')
    parser.add_argument('--train_val_split', type=float, default=0.5, help='ratio of train/val dataset split (between 0 and 1)')
    parser.add_argument('--num_test_points', type=int, default=50, help='Number of unique test points for cond. Wasserstein distance evaluation')
    parser.add_argument('--size_per_test_point', type=int, default=300, help='Number of repetitions for each test point')

    # Real dataset specific options
    parser.add_argument('--dataset_name', type=str, 
                        help='Name of real dataset as given in dataset_list.py. Also the name in the dataset folder')

    # WCGAN-specific hyperparameters
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--lambda_gp', type=float, default=2e-2)
    parser.add_argument('--one_sided', type=bool, default=True)

    args = parser.parse_args()
    config = vars(args)
    # Read additional config from file
    if args.config:
        assert os.path.exists(args.config), "No config file: {}".format(args.config)
        config_from_file = json_to_dict(args.config)
        # Make sure all options in config file also exist in argparse config.
        # Avoids choosing wrong parameters because of typos etc.
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(["Unknown option in config file: {}".format(opt)
            for opt in unknown_options])
        assert (not unknown_options), unknown_error

        config.update(config_from_file)

    # Read cgan architectures from config file
    spec = read_spec(config['architecture_dir'])
    return config, spec

def learning_cleanup(constants, config): 
    if config['scatter']:
        assert (constants['x_dim']==1), '"scatter" option only avaliable for X_DIM=1 and Y_DIM=1'

    PLOT_PATH = constants['plot_path']
    PLT_DATASET_NAME = constants['plt_dataset_name']
    FILE_NAME = constants['file_name']
    # clean up before training
    learning_savepath = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,config['prog_prefix'])
    if not os.path.exists(learning_savepath):
        os.makedirs(learning_savepath)
        print("{} created.".format(learning_savepath))
    else:
        try:
            with os.scandir(learning_savepath) as entries:
                for entry in entries:
                    if entry.is_file():
                        os.unlink(entry.path)
                    else:
                        shutil.rmtree(entry.path)
                print("All files and subdirectories in {} deleted successfully.".format(learning_savepath))
        except OSError:
            print("Error occurred while deleting files and subdirectories in {}.".format(learning_savepath))


    # if config["pdf_index"]:
    #     opt_value = config["pdf_index"]
    #     if opt_value:
    #         list_of_indexes = [int(s) for s in
    #                         opt_value.split(",")]
    #         for idx in list_of_indexes:
    #             learning_savepath = os.path.join(PLOT_PATH,PLT_DATASET_NAME,FILE_NAME,
    #                                              '{}_idx{}'.format(config['pdf_index_prefix'],idx))
    #             if not os.path.exists(learning_savepath):
    #                 os.makedirs(learning_savepath)
    #             else:
    #                 for f in os.listdir(learning_savepath):
    #                     os.remove(os.path.join(learning_savepath,f))
