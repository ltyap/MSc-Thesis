import numpy as np
import pandas as pd
import os
# Pre-processing data
from sklearn import model_selection as model
from sklearn.preprocessing import StandardScaler
import dataset_list 
import argparse


def generate(name):
    print('Dataset:',name)

    dataset = dataset_list.get_dataset_spec(name)()
    train_set, test_set = dataset.load_data()

    # Get scaling parameters using StandardScaler
    ss = StandardScaler().set_output(transform='pandas')
    ss.fit(dataset.scaling_ref)

    # Scale training data
    train_set_scaled = ss.transform(train_set)
    # Scale test data
    test_set_scaled = ss.transform(test_set)
    header= train_set.columns.values.tolist()

    # if separate (raw) validation dataset exists
    if os.path.exists(dataset.raw_val_data_path):
        train_split = train_set_scaled
        df_val = pd.read_csv(dataset.raw_val_data_path, header = 0, index_col = 0)
        # Make sure validation data is in same order as training data
        val = df_val.loc[:, header]
        # Scale validation data
        val_split = ss.transform(val)
    else:
        # get validation data by splitting original set into train and val
        train_split, val_split = model.train_test_split(train_set_scaled, test_size=0.2, random_state=42)

    # Check for dataset directory
    if not os.path.exists(dataset.dataset_save_path):
        os.makedirs(dataset.dataset_save_path)
    assert os.path.exists(dataset.dataset_save_path),("dataset folder {} does not exist".format(dataset.dataset_save_path))

    print("Dataset save path:", dataset.dataset_save_path)
    if dataset.channels != None:
        pd.DataFrame(train_split, columns=header)\
            .to_csv("{}/train.csv".format(dataset.dataset_save_path),header=True, index=False)
        pd.DataFrame(val_split, columns=header)\
            .to_csv("{}/val.csv".format(dataset.dataset_save_path),header=True, index=False)
        pd.DataFrame(test_set_scaled, columns=header)\
            .to_csv("{}/test.csv".format(dataset.dataset_save_path),header=True, index=False)
    else:
        np.savetxt("{}/train.csv".format(dataset.dataset_save_path), train_split,delimiter=",")
        np.savetxt("{}/val.csv".format(dataset.dataset_save_path), val_split,delimiter=",")
        np.savetxt("{}/test.csv".format(dataset.dataset_save_path), test_set_scaled,delimiter=",")
    
    return dataset, test_set_scaled

# # Plotting
def plotting(dataset, test_set_scaled):
    dataset.plot_test_data(test_set_scaled)

def main():
    parser = argparse.ArgumentParser(description='Preprocess real-world datasets')
    parser.add_argument("--dataset", type=str, help='Name of dataset to preprocess')
    parser.add_argument("--scatter", type=bool, help='Scatter/KDE plots of datasets', default=False)
    args = parser.parse_args()
    if args.dataset == "all":
        for ds_name in dataset_list.set:
            dataset, scaled_test_set = generate(ds_name)
            if args.scatter:
                plotting(dataset, scaled_test_set)                
    else:
        assert args.dataset in dataset_list.set, (
                "Unknown dataset: {}".format(args.dataset))
        dataset, scaled_test_set = generate(args.dataset)
        if args.scatter:
            plotting(dataset, scaled_test_set)
          
if __name__ == "__main__":
    main()