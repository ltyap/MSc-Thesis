from dataset_specifications.dataset import Dataset
import dataset_specifications.dataset as dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class AeroSet(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'aero'
        self.x_dim = 3
        self.y_dim = 1
        self.synthetic = False        
        self.val_path = None
        self.channels = ['TwrBsMyt_[kN-m] mean', 'TwrBsMyt_[kN-m] max', 'TwrBsMyt_[kN-m] stddev','TwrBsMyt_[kN-m] ST_DEL', 
                         'RootMyb1_[kN-m] mean', 'RootMyb1_[kN-m] max', 'RootMyb1_[kN-m] stddev','RootMyb1_[kN-m] ST_DEL',
                         'RootMxb1_[kN-m] mean', 'RootMxb1_[kN-m] max', 'RootMxb1_[kN-m] stddev', 'RootMxb1_[kN-m] ST_DEL',
                         'YawBrMyn_[kN-m] mean', 'YawBrMyn_[kN-m] max', 'YawBrMyn_[kN-m] stddev', 'YawBrMyn_[kN-m] ST_DEL']
        
        self.inputs = ['URef','PLExp','IECturbc']

        # self.channel_name = 'YawBrMyn_[kN-m] ST_DEL'
        self.key = {
            "TwrBsMyt_[kN-m] mean": "TwrBsMyt_mean",
            "TwrBsMyt_[kN-m] max": "TwrBsMyt_max",
            'TwrBsMyt_[kN-m] stddev': "TwrBsMyt_stddev",
            'TwrBsMyt_[kN-m] ST_DEL':"TwrBsMyt_ST_DEL", 
            'RootMyb1_[kN-m] mean':"RootMyb1_mean",
            'RootMyb1_[kN-m] max': "RootMyb1_max",
            'RootMyb1_[kN-m] stddev': "RootMyb1_stddev",
            'RootMyb1_[kN-m] ST_DEL': "RootMyb1_ST_DEL",
            'RootMxb1_[kN-m] mean': "RootMxb1_mean",
            'RootMxb1_[kN-m] max': "RootMxb1_max",
            'RootMxb1_[kN-m] stddev': "RootMxb1_stddev",
            'RootMxb1_[kN-m] ST_DEL': "RootMxb1_ST_DEL",
            'YawBrMyn_[kN-m] mean': "YawBrMyn_mean",
            'YawBrMyn_[kN-m] max': "YawBrMyn_max",
            'YawBrMyn_[kN-m] stddev': "YawBrMyn_stddev",
            'YawBrMyn_[kN-m] ST_DEL': "YawBrMyn_ST_DEL"
        }
        self.dataset_save_path = os.path.join("datasets",self.name)

    def load_data(self):
        df_train = pd.read_csv("datasets/{}/raw_data/train/data_raw.dat".format(self.name), header = 0, index_col = 0)
        df_test = pd.read_csv("datasets/{}/raw_data/test/data_raw.dat".format(self.name), header = 0, index_col = 0)

        # train = df_train[["URef", "PLExp", "IECturbc", self.channel_name]]
        # test = df_test.loc[:, ["URef", "PLExp", "IECturbc", self.channel_name]]
        tmp = self.inputs+self.channels
        train = df_train.loc[:, tmp]
        test = df_test.loc[:, tmp]
        self.train_set = train
        self.test_set = test

        path = "datasets/aero/raw_data/train/data_raw.dat"
        print("Path to scaling data:", path)
        scaling_ref = pd.read_csv(path, header = 0, index_col = 0) # ----------------------------> update training data path here (used only for scaling)
        self.scaling_ref = scaling_ref.loc[:, tmp]

        return train, test
    def plot_test_data(self, test_data_scaled):
        for channel_name in self.channels:
            channel = self.key[channel_name]
            plot_path = './plots/{}/{}'.format(self.name, channel)
            plot_name = 'Scatter_Test_scaled'
            # Initialise directory
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            for input in self.inputs:
                plt.figure()
                plt.scatter(test_data_scaled[input],test_data_scaled[channel_name], c='k', s=4)
                plt.xlabel(input)
                plt.ylabel(channel)
                plt.savefig(os.path.join(plot_path, '{}_{}'.format(plot_name,input)))
                plt.close()

            print(self.inputs+[channel_name])
            # print(test_data_scaled.head())
            test_data_scaled_selection = test_data_scaled.loc[:,self.inputs+[channel_name]]

            test_scaled_LD = dataset.LabelledData(x= test_data_scaled_selection.to_numpy()[:,:self.x_dim],
                                            y = test_data_scaled_selection.to_numpy()[:,self.x_dim:])
            test_LD = dataset.LabelledData(x= self.test_set.to_numpy()[:,:self.x_dim],
                                        y = self.test_set.to_numpy()[:,self.x_dim:])

            plt_type = 'Test_PDF'
            testpdf_plot_path = os.path.join(plot_path,plt_type)

            if not os.path.exists(testpdf_plot_path):
                os.makedirs(testpdf_plot_path)
            else:
                for f in os.listdir(testpdf_plot_path):
                    os.remove(os.path.join(testpdf_plot_path,f))
            assert os.path.exists(testpdf_plot_path),("dataset folder {} does not exist".format(testpdf_plot_path))

            x, idx, counts = np.unique(test_scaled_LD.x, return_counts = True, return_index = True, axis = 0)
            tmp = np.argsort(idx)
            idx = idx[tmp]
            x = x[tmp]
            counts = counts[tmp]
            
            start_idx = idx
            end_idx = idx+counts
            for _, (start, end) in enumerate(zip(start_idx,end_idx)):
                plt.figure()
                sns.kdeplot(test_scaled_LD.y[start:end].squeeze(), color='k')
                plt.title('x={}, idx = {}-{}'.format(test_LD.x[start], start, end-1), fontsize=10)
                plt.savefig('{}/idx_{}-{}.png'.format(testpdf_plot_path, start, end-1))
                plt.close()