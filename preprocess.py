import numpy as np
import os
# Pre-processing data
from sklearn import model_selection as model
from sklearn.preprocessing import StandardScaler
import dataset_list 

name = 'aerohydro'
print(name)

dataset = dataset_list.get_dataset_spec(name)()
train_set, test_set = dataset.load_data()

ss = StandardScaler().set_output(transform='pandas')
train_set_scaled = ss.fit_transform(train_set)
# Scale test data based on training data
test_set_scaled = ss.transform(test_set)

train_split, val_split = model.train_test_split(train_set_scaled, test_size=0.3, random_state=42)

# Check for dataset directory
if not os.path.exists(dataset.dataset_save_path):
    os.makedirs(dataset.dataset_save_path)
assert os.path.exists(dataset.dataset_save_path),("dataset folder {} does not exist".format(dataset.dataset_save_path))

print(dataset.dataset_save_path)
np.savetxt("{}/train.csv".format(dataset.dataset_save_path), train_split,delimiter=",")
np.savetxt("{}/val.csv".format(dataset.dataset_save_path), val_split,delimiter=",")
np.savetxt("{}/test.csv".format(dataset.dataset_save_path), test_set_scaled,delimiter=",")


dataset.plot_test_data(test_set_scaled)