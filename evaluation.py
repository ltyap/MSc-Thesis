from misc_funcs import indexes
import torch
from dataset_specifications.dataset import LabelledData
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import ot

def evaluate_model(model, data, data_train, data_test=None, epoch=None, make_plots = True):
    """
    data: validation data
    """
    testing = (epoch==None) #returns False/0 if epoch is not None
    config = model.config
    evaluation_vals = model.eval(data, kde_eval, use_best_kernel_scale=testing)
    # Calculate Wasserstein-2 dist w.r.t. validation data
    w2_dist = wasserstein_from_samples(model, data)
    evaluation_vals['Wasserstein-2 dist'] = w2_dist
    val_method = "true"
    metric_string = "\t".join(["{}: {:.5}".format(key, validation_val) for
        (key,validation_val) in evaluation_vals.items()])
    print("Epoch {}, {}\t{}".format(epoch, val_method, metric_string))

    tmp = data.x.shape[0]
    data_train_partial = LabelledData(x=data_train.x[:tmp], y=data_train.y[:tmp])
    evaluation_vals_train = model.eval(data_train_partial, kde_eval, use_best_kernel_scale = testing)
    metric_string = "\t".join(["{}: {:.5}".format(key, training_val) for
        (key,training_val) in evaluation_vals_train.items()])
    # print("Training Epoch {}, {}\t{}".format(epoch, val_method, metric_string))

    if make_plots:
        if config["scatter"]:
            model_samples = model.sample(data.x, batch_size = config["eval_batch_size"])
            if type(model_samples) == torch.Tensor:
                model_samples = model_samples.to("cpu")
            model_data = LabelledData(x=data.x, y = model_samples)
            plot_title = 'Training epoch {}'.format(epoch)
            sample_sets = [data, model_data]
            labels = ["Ground Truth", "model"]
            plot_path = os.path.join(model.constants['plot_path'],model.constants['plt_dataset_name'], model.constants['file_name'])
 
            plot_samples(sample_sets, file_name = epoch, path_name = plot_path,
                        labels = labels, title=plot_title, range_dataset=data)
        if config["pdf_index"]:
            opt_value = config["pdf_index"]
            if opt_value:
                list_of_indexes = [int(s) for s in
                            opt_value.split(",")]
                for idx in list_of_indexes:
                    tmp = indexes(data_test.x[idx], data_test.x)
                    model_samples = model.sample(data_test.x[tmp])
                    if type(model_samples) == torch.Tensor:
                        model_samples = model_samples.to("cpu")
                    plt.figure()
                    sns.kdeplot(model_samples.squeeze(), color='b', label='Gen')
                    sns.kdeplot(data_test.y[tmp].squeeze(), color='k', linestyle='--', label='True')
                    title = "Epoch {}, x = {}".format(epoch,data_test.x[idx].to('cpu'))
                    plt.title(title, fontsize=10)
                    plt.legend()
                    savepath = os.path.join(model.plots_path,'learning_prog_idx{}'.format(idx))
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    assert os.path.exists(savepath),("dataset folder {} does not exist".format(savepath))
                    plt.savefig(os.path.join(savepath,'{}.png'.format(epoch)))
                    plt.close()

    return evaluation_vals, evaluation_vals_train

def mae_from_samples(samples, y):
    medians,_ = torch.median(samples, dim=1)
    abs_errors = torch.sum(torch.abs(medians-y), dim = 1)
    return torch.mean(abs_errors).item()


def wasserstein_from_samples(model, data):
    # Calculate average Wasserstein-2 distance between
    # generated samples and real samples from validation set
    x = data.x.to(model.device)
    y = data.y.detach().numpy()
    samples = model.sample(x, batch_size=model.config["eval_batch_size"])
    M_wasserstein = ot.wasserstein_1d(samples.cpu().detach().numpy(), y, p = 2)**0.5
    y_std = np.std(y, axis=0)
    M_wasserstein_normalised = M_wasserstein/y_std
    return M_wasserstein_normalised.mean()

def kde_eval(model, data, kernel_scale=None):
    x = data.x.to(model.device)
    x_repeated = torch.repeat_interleave(x, repeats = model.config["eval_samples"], dim = 0)
    shape_y = data.y.shape
    # samples_wasserstein = model.sample(x, batch_size = model.config["eval_batch_size"])
    samples = model.sample(x_repeated, batch_size = model.config["eval_batch_size"]).reshape(
        shape_y[0],model.config["eval_samples"],shape_y[1])
    y = data.y.to(model.device)
    diff = samples - y.unsqueeze(1)
    squarednorm = torch.sum(torch.pow(diff,2), dim = 2)

    if not (kernel_scale == None):
        n_h = 1
        h_squared = torch.tensor([kernel_scale], device = model.device)
    else:
        n_h = model.config["kernel_scales"]
        h_squared = torch.logspace(
            math.log10(model.config["kernel_scale_min"]),
            math.log10(model.config["kernel_scale_max"]), steps = n_h)
    h_squared = h_squared.to(model.device)
    scale_lls = []
    #Batch over kernel scales
    h_squared_splits = torch.split(h_squared, model.config["kde_batch_size"], dim = 0)

    for splits in h_squared_splits:
        normalisation = torch.pow(torch.rsqrt(2.*math.pi*splits), model.constants['y_dim'])
        ratio = squarednorm.unsqueeze(dim=2).repeat((1,1,splits.shape[0]))/splits

        ll_y = torch.log(normalisation)+torch.logsumexp(-0.5*(ratio), dim = 1)-\
                math.log(model.config["eval_samples"])
        mean_lls = torch.mean(ll_y, dim = 0)
        scale_lls.append(mean_lls)
    joined_scale_lls = torch.cat(scale_lls, dim = 0)

    argmax = torch.argmax(joined_scale_lls)
    best_scale = h_squared[argmax]

    best_ll = joined_scale_lls[argmax].item()
    best_mae = mae_from_samples(samples, y)
    # wasserstein_dist = wasserstein_from_samples(samples_wasserstein,y)
    evaluation_vals = {
        "ll": best_ll,
        "mae": best_mae,
    }

    return evaluation_vals, best_scale


def plot_samples(sample_sets, file_name, path_name, labels=None, range_dataset=None,
        title=None, both_y=False):
    n_sets = len(sample_sets)
    fig, axes = plt.subplots(nrows=1, ncols=n_sets, squeeze=False,
            figsize=(7.*n_sets, 4.5))
    x_range = (min(range_dataset.x), max(range_dataset.x))
    y_range = (min(range_dataset.y), max(range_dataset.y))
    for set_i, (samples, ax) in enumerate(zip(sample_sets, axes[0])):
        ax.scatter(samples.x, samples.y, s=3, color="green")

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        if labels:
            ax.set(title=labels[set_i])
        if both_y:
            ax.set(xlabel="$y_1$", ylabel="$y_2$")
        else:
            ax.set(xlabel="$x$", ylabel="$y$")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if not os.path.exists(path_name):
            os.makedirs(path_name)
    assert os.path.exists(path_name),("model parameter folder {} does not exist".format(path_name))
    plt.savefig('{}/{}.png'.format(path_name, file_name))
    plt.close(fig)
