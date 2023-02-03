import torch


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