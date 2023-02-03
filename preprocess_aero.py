import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import collections

# Pre-processing aero data
import pandas as pd
from sklearn import model_selection as model
from misc_funcs import indexes


LabelledData = collections.namedtuple("LabelledData",["x","y"])

X_DIM = 3
Y_DIM = 1

df_train = pd.read_csv("datasets/aero/raw_data/train/data_raw.dat", header = 0, index_col = 0)
aero_train = df_train[["URef", "PLExp", "IECturbc", "TwrBsMyt_[kN-m] ST_DEL"]]
aero_train_mean = aero_train.mean()
aero_train_std = aero_train.std()
aero_train_scale = (aero_train - aero_train_mean)/aero_train_std

aero_train_split, aero_val_split = model.train_test_split(aero_train_scale, test_size=0.3, random_state=42)
# np.savetxt("train.csv", aero_train_split,delimiter=",")
# np.savetxt("val.csv", aero_val_split,delimiter=",")


# Test Data
df_test = pd.read_csv("datasets/aero/raw_data/test/data_raw.dat", header = 0, index_col = 0)

PLOT_PATH = './plots/aero_TwrBsMyt_ST_DEL'
PLOT_NAME = 'Scatter_Test_Uref'
# Initialise directory
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)

plt.figure()
plt.scatter(df_test["URef"],df_test["TwrBsMyt_[kN-m] ST_DEL"])
plt.xlabel('U_ref')
plt.ylabel('TwrBsMyt_ST_DEL')
plt.savefig(os.path.join(PLOT_PATH, PLOT_NAME))
plt.close()


# Scale test data based on training data
aero_test_raw = df_test.loc[:, ["URef", "PLExp", "IECturbc", "TwrBsMyt_[kN-m] ST_DEL"]]
aero_test_scale = (aero_test_raw-aero_train_mean)/aero_train_std

# Convert to 'LabelledData' datatype
test_scale = LabelledData(x= aero_test_scale.to_numpy()[:,:X_DIM],y = aero_test_scale.to_numpy()[:,X_DIM:])
test_raw = LabelledData(x= aero_test_raw.to_numpy()[:,:X_DIM],y = aero_test_raw.to_numpy()[:,X_DIM:])


listofx = np.linspace(0,14700,50, dtype=int)
PLT_TYPE = 'Test_PDF'
testpdf_plot_path = os.path.join(PLOT_PATH,PLT_TYPE)

if not os.path.exists(testpdf_plot_path):
    os.makedirs(testpdf_plot_path)
else:
    for f in os.listdir(testpdf_plot_path):
        os.remove(os.path.join(testpdf_plot_path,f))

assert os.path.exists(testpdf_plot_path),("dataset folder {} does not exist".format(testpdf_plot_path))

for _, idx in enumerate(listofx):
    tmp = indexes(test_scale.x[idx], test_scale.x)
    plt.figure()
    sns.kdeplot(test_scale.y[tmp].squeeze(), color='k')
    plt.title('x={}, idx = {}'.format(test_raw.x[idx], idx), fontsize=10)

    plt.savefig('{}/idx_{}.png'.format(testpdf_plot_path, idx))
    plt.close()
