import numpy as np
import pandas as pd
import os
# Pre-processing data
from sklearn import model_selection as model
from sklearn.preprocessing import StandardScaler
import dataset_list 

name = 'aero'
print('Dataset:',name)

dataset = dataset_list.get_dataset_spec(name)()
train_set, test_set = dataset.load_data()

ss = StandardScaler().set_output(transform='pandas')
ss.fit(dataset.scaling_ref)
train_set_scaled = ss.transform(train_set)
# Scale test data based on scaling reference
test_set_scaled = ss.transform(test_set)

train_split, val_split = model.train_test_split(train_set_scaled, test_size=0.2, random_state=42)

# Check for dataset directory
if not os.path.exists(dataset.dataset_save_path):
    os.makedirs(dataset.dataset_save_path)
assert os.path.exists(dataset.dataset_save_path),("dataset folder {} does not exist".format(dataset.dataset_save_path))

print("Dataset save path:", dataset.dataset_save_path)
if dataset.channels != None:
    train_split_df = pd.DataFrame(train_split, columns=dataset.inputs+dataset.channels).to_csv("{}/train.csv".format(dataset.dataset_save_path),header=True, index=False)
    val_split_df = pd.DataFrame(val_split, columns=dataset.inputs+dataset.channels).to_csv("{}/val.csv".format(dataset.dataset_save_path),header=True, index=False)
    test_set_scaled_df = pd.DataFrame(test_set_scaled, columns=dataset.inputs+dataset.channels).to_csv("{}/test.csv".format(dataset.dataset_save_path),header=True, index=False)
else:
    np.savetxt("{}/train.csv".format(dataset.dataset_save_path), train_split,delimiter=",")
    np.savetxt("{}/val.csv".format(dataset.dataset_save_path), val_split,delimiter=",")
    np.savetxt("{}/test.csv".format(dataset.dataset_save_path), test_set_scaled,delimiter=",")

# # Plotting
dataset.plot_test_data(test_set_scaled)