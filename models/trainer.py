import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
from torch_geometric.data import Data

import random
import pandas as pd
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU, BatchNorm1d, Softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import DataLoader

import torch.optim as optim
from tqdm import trange


#Define the processor layer, which does the message passing
class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, nlp_hidden_dim, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nlp_hidden_dim = nlp_hidden_dim

        #Build the node NLP
        self.node_nlp = Sequential(Linear(2 * self.in_channels, self.nlp_hidden_dim), 
                                   ReLU(), 
                                   Linear(self.nlp_hidden_dim, self.out_channels),
                                   LayerNorm(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.node_nlp[0].reset_parameters()
        self.node_nlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        #x is the node features
        #edge_index is the edge indices
        #edge_attr is the edge features

        #Calculate the edge messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size) #shape: [num_edges, out_channels]

        #Calculate the node messages using aggregated messages and self embedding
        out = self.node_nlp(torch.cat([x, out], dim=1))

        return out

    def message(self, x_j, edge_attr):
        #x_j is the node features of the neighboring nodes
        #edge_attr is the edge feature

        #Calculate the edge messages
        out = x_j * edge_attr
        return out

    def aggregate(self, out, edge_index, dim_size=None):
        #out is the edge messages
        #edge_index is the edge indices

        #The axis along which to index the number of nodes
        node_dim = 0

        #Aggregate the edge messages
        out = torch_scatter.scatter(out, edge_index[0, :], dim=node_dim, reduce='mean')
        return out



#Building the graph neural network model
class neuralGNN(torch.nn.Module):
    def __init__(self, time_window_size, proc_nlp_hidden_dim, time_nlp_hidden_dim,
                 num_supernodes, super_nlp_hidden_dim_1, super_nlp_hidden_dim_2,
                 num_layers):
        super(neuralGNN, self).__init__()

        self.time_window_size = time_window_size
        self.proc_nlp_hidden_dim = proc_nlp_hidden_dim
        self.time_nlp_hidden_dim = time_nlp_hidden_dim
        self.num_supernodes = num_supernodes
        self.super_nlp_hidden_dim_1 = super_nlp_hidden_dim_1
        self.super_nlp_hidden_dim_2 = super_nlp_hidden_dim_2
        self.num_layers = num_layers

        #Build the graph processing layers
        self.processor = nn.ModuleList()
        assert self.num_layers > 0

        processor_layer = self.buildProcessorModel()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(self.time_window_size, 
                                                  self.time_window_size,
                                                  self.proc_nlp_hidden_dim))

        #Define the time compression NLP
        self.time_compress_mlp = Sequential(Linear(self.time_window_size, self.time_nlp_hidden_dim),
                                            ReLU(),
                                            Linear(self.time_nlp_hidden_dim, 1),
                                            LayerNorm(1))

        #Define the supernode NLP
        self.supernode_mlp = Sequential(Linear(self.num_supernodes, self.super_nlp_hidden_dim_1),
                                        #BatchNorm1d(self.super_nlp_hidden_dim_1),
                                        ReLU(),
                                        Linear(self.super_nlp_hidden_dim_1, self.super_nlp_hidden_dim_2),
                                        #BatchNorm1d(self.super_nlp_hidden_dim_2),
                                        ReLU(),
                                        Linear(self.super_nlp_hidden_dim_2, 1),
                                        Softmax(dim=1))


    def buildProcessorModel(self):
        return ProcessorLayer

    def forward(self, data, supernode_indices):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        #Step 1: Process the graph
        for i in range(self.num_layers):
            x = self.processor[i](x=x, edge_index=edge_index, edge_attr=edge_attr)

        #Step 2: Time compression
        x = self.time_compress_mlp(x)

        #Step 3: Supernode aggregation
        #NOTE: Check that the supernodes are concatenated into a vector for processing by the supernode mlp
        supernodes = x[supernode_indices]

        pred = self.supernode_mlp(supernodes.T)

        return pred



def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



def train(dataset, device, args):
    #Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #Build the data loader
    #data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data = dataset.to(device)


    #Find the supernode indices
    random_indices = np.random.choice(data.edge_index.shape[-1], size=args.num_supernodes, replace=False)
    supernode_indices = data.edge_index[0,random_indices]

    #remove duplicate nodes
    supernode_indices = np.unique(supernode_indices.cpu().numpy())

    final_num_supernodes = len(supernode_indices)

    print("Number of supernodes: %d" % final_num_supernodes)



    #Build the model
    model = neuralGNN(time_window_size=args.time_window_size,
                      proc_nlp_hidden_dim=args.proc_nlp_hidden_dim,
                      time_nlp_hidden_dim=args.time_nlp_hidden_dim,
                      num_supernodes=final_num_supernodes,
                      super_nlp_hidden_dim_1=args.super_nlp_hidden_dim_1,
                      super_nlp_hidden_dim_2=args.super_nlp_hidden_dim_2,
                      num_layers=args.num_layers).to(device)

    #Build the optimizer
    scheduler, optimizer = build_optimizer(args, model.parameters())

    #Build the loss function
    #loss_fn = nn.NLLLoss()
    loss_fn = nn.MSELoss()


    #Train the model
    for epoch in range(args.epochs):
        model.train()

        #for data in data_loader:
        #data = data.to(device)
        optimizer.zero_grad()
        out = model(data, supernode_indices)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss.item()))


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

for args in [
        {
        'time_window_size': 21,
        'proc_nlp_hidden_dim': 32,
        'time_nlp_hidden_dim': 32,
        'num_supernodes': 500,
        'super_nlp_hidden_dim_1': 128,
        'super_nlp_hidden_dim_2': 32,
        'num_layers': 5,
        'batch_size': 1,
        'epochs': 5000,
        'opt': 'adam',
        'opt_scheduler': 'none',
        'lr': 0.001,
        'device': 'cuda',
        'seed': 42,
        'weight_decay': 0.0005,
        },
    ]:
        args = objectview(args)



dataset = torch.load('/workspace/data_gen/pupil_direction_graphs.pt')[0]

dataset.y = torch.tensor([0], dtype=torch.float).unsqueeze(-1)
# for i, data in enumerate(dataset):
#     print(data.y.item())
#     # if(np.isclose(data.y.item(), 0.0)):
#     #     print("Index: %d" % i)
# import pdb;pdb.set_trace()
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
args.device = device
print("device in use: {}".format(device))




#Train the model
train(dataset, device, args)