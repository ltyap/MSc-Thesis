import torch
import torch.distributions as tdists

class CensoredNormal():         
    def __init__(self, config, device):
        self.tmp = tdists.multivariate_normal.MultivariateNormal(
            torch.zeros(config["noise_dim"], device=device),
            torch.eye(config["noise_dim"], device=device)
        )
    def sample(self, size):
        samples = self.tmp.sample(size)
        return torch.max(samples, torch.zeros_like(samples))

class LogNormal():         
    def __init__(self, config, device):
        self.tmp = tdists.multivariate_normal.MultivariateNormal(
            torch.zeros(config["noise_dim"], device=device),
            torch.eye(config["noise_dim"], device=device)
        )
    def sample(self, size):
        samples = self.tmp.sample(size)
        return torch.exp(samples)


def get_gaussian(config, device):
    return tdists.multivariate_normal.MultivariateNormal(
            torch.zeros(config["noise_dim"], device=device),
            torch.eye(config["noise_dim"], device=device)
        ) #isotropic

def get_uniform(config, device):
    return tdists.uniform.Uniform(
            torch.zeros(config["noise_dim"], device=device),
            torch.ones(config["noise_dim"], device=device)
        ) # Uniform on [0,1]

def get_exponential(config, device):
    return tdists.exponential.Exponential(
            torch.ones(config["noise_dim"], device=device)
        ) # Exponential, rate 1

def get_lognormal(config, device):
    return LogNormal(config, device)
def get_censored(config, device):
    return CensoredNormal(config, device) 

noise_dists = {
    "gaussian": get_gaussian,
    "uniform": get_uniform,
    "exponential": get_exponential,
    "lognormal": get_lognormal,
    "censorednormal":get_censored
}

def get_noise_dist(config, device):
    dist = config["noise_dist"]
    assert (dist in noise_dists), "Unknown noise distribution: {}".format(dist)
    return noise_dists[dist](config, device)

