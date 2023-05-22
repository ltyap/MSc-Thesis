from dataset_specifications.dataset import Dataset
import dataset_specifications.dataset as dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Floating(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'floating'
        self.x_dim = 7
        self.y_dim = 1
        self.synthetic = False        
        self.val_path = "datasets/{}/raw_data/train/validation/data_raw.dat".format(self.name)
        self.channels = ['Mt_x_1_1_mean',
       'Mt_y_1_1_mean', 'Mt_x_14_2_mean', 'Mt_y_14_2_mean', 'Mb_x_1_1_mean',
       'Mb_y_1_1_mean', 'Fl_m1br1_EfTn_mean', 'Fl_m1br2_EfTn_mean',
       'Fl_m2br1_EfTn_mean', 'Fl_m2br2_EfTn_mean', 'Fl_m3br1_EfTn_mean',
       'Fl_m3br2_EfTn_mean', 'Roll_OF_mean', 'Pitch_OF_mean', 'Yaw_OF_mean',
       'uB1_x_17_mean', 'uB1_y_17_mean', 'uB1_z_17_mean', 'Mt_x_1_1_max',
       'Mt_y_1_1_max', 'Mt_x_14_2_max', 'Mt_y_14_2_max', 'Mb_x_1_1_max',
       'Mb_y_1_1_max', 'uaT_x_15_max', 'uaT_y_15_max', 'uaT_z_15_max',
       'Fl_m1br1_EfTn_max', 'Fl_m1br2_EfTn_max', 'Fl_m2br1_EfTn_max',
       'Fl_m2br2_EfTn_max', 'Fl_m3br1_EfTn_max', 'Fl_m3br2_EfTn_max',
       'Roll_OF_max', 'Pitch_OF_max', 'Yaw_OF_max', 'uB1_x_17_max',
       'uB1_y_17_max', 'uB1_z_17_max', 'Mt_y_1_1_stel',
        'Mt_y_14_2_stel', 'Mb_x_1_1_stel', 'Mb_y_1_1_stel', 'Mt_x_1_1_stel',
       'Mt_x_14_2_stel']
        
        self.inputs = ['URef','PLExp','TI','Hs','Tp','Wdir','Yaw']

        self.dataset_save_path = os.path.join("datasets",self.name)

    def load_data(self):
        df_train = pd.read_csv("datasets/{}/raw_data/train/data_raw.csv".format(self.name), header=0)
        df_test = pd.read_csv("datasets/{}/raw_data/test/data_raw.csv".format(self.name), header=0)
        tmp = self.inputs+self.channels
        train = df_train.loc[:, tmp]
        test = df_test.loc[:, tmp]
        self.train_set = train
        self.test_set = test

        path = "datasets/{}/raw_data/train/data_raw.csv".format(self.name)
        print("Path to scaling data:", path)
        scaling_ref = pd.read_csv(path, header = 0) # ----------------------------> update training data path here (used only for scaling)
        self.scaling_ref = scaling_ref.loc[:, tmp]

        return train, test
    def plot_test_data(self, test_data_scaled):
        for channel_name in self.channels:
            # channel = self.key[channel_name]
            plot_path = './plots/{}/{}'.format(self.name, channel_name)
            plot_name = 'Scatter_Test_scaled'
            # Initialise directory
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            for input in self.inputs:
                plt.figure()
                plt.scatter(test_data_scaled[input],test_data_scaled[channel_name], c='k', s=4)
                plt.xlabel(input)
                plt.ylabel(channel_name)
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