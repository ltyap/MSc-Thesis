import torch.nn as nn
import torch

#create NN classes for discrimator/generator
def ffsection(x_dim, other_dim, layer_list, extra_in_hidden_dim=0):
    if not layer_list:
        layers = []
        out_dim = x_dim+extra_in_hidden_dim
    else:
        layers = [nn.Linear(x_dim + other_dim, layer_list[0])]+\
                [nn.Linear(from_dim+extra_in_hidden_dim, to_dim) \
                for from_dim, to_dim in zip(layer_list[:-1], layer_list[1:])]

    out_dim = layer_list[-1]
    return nn.ModuleList(layers), out_dim

class NoiseInjection(nn.Module):
    def __init__(self, nn_spec) -> None:
        super().__init__()
        self.activation = nn_spec["activation"]
        self.nodes_per_layer = nn_spec["nodes_per_layer"]
        self.other_dim = nn_spec["other_dim"]
        self.cond_dim = nn_spec["cond_dim"]
        self.output_dim = nn_spec["output_dim"]
        
        self.layers, self.last_layer_dim = ffsection(
            self.cond_dim, self.other_dim, self.nodes_per_layer, self.other_dim)
        self.output_layer = nn.Linear((self.last_layer_dim+self.other_dim),self.output_dim)
    def forward(self, x):
        hidden_repr = x[:,:self.cond_dim]
        noise = x[:,self.cond_dim:]
        for layer in self.layers:
            combined_repr = torch.cat((hidden_repr, noise), dim = 1)
            hidden_repr = self.activation(layer(combined_repr))
        hidden_repr = torch.cat((hidden_repr, noise),dim =1)
        return self.output_layer(hidden_repr)
class FeedForward(nn.Module):
    def __init__(self, nn_spec) -> None:
        super().__init__()
        self.activation = nn_spec["activation"]
        self.nodes_per_layer = nn_spec["nodes_per_layer"]
        self.other_dim = nn_spec["other_dim"]
        self.cond_dim = nn_spec["cond_dim"]
        self.output_dim = nn_spec["output_dim"]
        
        self.layers, self.last_layer_dim = ffsection(
            self.cond_dim, self.other_dim, self.nodes_per_layer)

        self.output_layer = nn.Linear(self.last_layer_dim,self.output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        return self.output_layer(x)#no activation fnc before output
class DoubleInputNetwork(nn.Module):
    def __init__(self, nn_spec) -> None:
        super().__init__()

        self.activation = nn_spec["activation"]
        self.dim_x = nn_spec["cond_dim"]

        self.cond_layers, cond_dim = ffsection(nn_spec["cond_dim"], other_dim= 0,
                                               layer_list= nn_spec["cond_layers"])
        self.other_layers, other_dim = ffsection(x_dim = 0,other_dim = nn_spec["other_dim"],
                                                 layer_list=nn_spec["other_layers"])
        self.hidden_layers, hidden_dim = ffsection(cond_dim, other_dim,
                                                    nn_spec["nodes_per_layer"])
       
        self.output_layer = nn.Linear(hidden_dim, nn_spec["output_dim"])


    def forward(self, x):
        cond_repr = x[:,:self.dim_x]
        other_repr = x[:,self.dim_x:]

        #conditional input
        for layer in self.cond_layers:
            cond_repr = self.activation(layer(cond_repr))
        #other (noise/real data)
        for layer in self.other_layers:
            other_repr = self.activation(layer(other_repr))

        hidden_input = torch.cat((cond_repr, other_repr), dim = 1)
        for layer in self.hidden_layers:
            hidden_input = self.activation(layer(hidden_input))
        output = self.output_layer(hidden_input)
        return output
