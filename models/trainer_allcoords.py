import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
from torch_geometric.data import Data

import random
import pandas as pd
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU, BatchNorm1d, Softmax, LeakyReLU
from torch.utils.data import random_split
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

import torch.optim as optim
from tqdm import trange

import args


#Define the processor layer, which does the message passing
class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, nlp_hidden_dim, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nlp_hidden_dim = nlp_hidden_dim

        #Build the node NLP
        self.node_nlp = Sequential(Linear(2 * self.in_channels, self.nlp_hidden_dim), 
                                   LeakyReLU(), 
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
                                            BatchNorm1d(self.time_nlp_hidden_dim),
                                            LeakyReLU(),
                                            Linear(self.time_nlp_hidden_dim, 1))

        #Define the supernode NLP
        self.supernode_mlp = Sequential(Linear(self.num_supernodes, self.super_nlp_hidden_dim_1),
                                        BatchNorm1d(self.super_nlp_hidden_dim_1),
                                        LeakyReLU(),
                                        Linear(self.super_nlp_hidden_dim_1, self.super_nlp_hidden_dim_2),
                                        BatchNorm1d(self.super_nlp_hidden_dim_2),
                                        LeakyReLU(),
                                        Linear(self.super_nlp_hidden_dim_2, 2))


    def buildProcessorModel(self):
        return ProcessorLayer

    def forward(self, data, supernode_indices, device):
        x, edge_index, edge_attr, batch_mask = data.x, data.edge_index, data.edge_attr, data.batch

        #Step 1: Process the graph
        for i in range(self.num_layers):
            x = self.processor[i](x=x, edge_index=edge_index, edge_attr=edge_attr)

        #Step 2: Time compression
        x = self.time_compress_mlp(x)

        batch_size = batch_mask[-1].item() + 1
        num_nodes_per_batch = torch.zeros((batch_size))
        #Count the number of nodes in each batch and ensure they are equal
        for i in range(batch_size):
            num_nodes_per_batch[i] = torch.sum(batch_mask == i)
        assert torch.all(num_nodes_per_batch == num_nodes_per_batch[0])

        #Reshape the time compressed node features into a 2D tensor
        x = x.reshape((int(batch_size), int(num_nodes_per_batch[0])))

        #Step 3: Supernode aggregation
        #NOTE: Check that the supernodes are concatenated into a vector for processing by the supernode mlp
        supernodes = x[:, supernode_indices]

        #import pdb; pdb.set_trace()
        pred = self.supernode_mlp(supernodes)

        #Apply sigmoid for binary classification
        #pred = torch.sigmoid(pred)

        return pred



def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay, amsgrad=True)
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



def train(dataset, supernode_indices, device, args):
    #Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #Build the data loader
    #Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    #data = dataset.to(device)



    #Build the model
    model = neuralGNN(time_window_size=args.time_window_size,
                      proc_nlp_hidden_dim=args.proc_nlp_hidden_dim,
                      time_nlp_hidden_dim=args.time_nlp_hidden_dim,
                      num_supernodes=len(supernode_indices),
                      super_nlp_hidden_dim_1=args.super_nlp_hidden_dim_1,
                      super_nlp_hidden_dim_2=args.super_nlp_hidden_dim_2,
                      num_layers=args.num_layers).to(device)

    #Build the optimizer
    scheduler, optimizer = build_optimizer(args, model.parameters())

    #Build the loss function
    #loss_fn = nn.NLLLoss()
    loss_fn = nn.MSELoss()
    #loss_fn = nn.CrossEntropyLoss()

    #Define a pandas dataframe to store the training results
    df = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'test_loss', 'test_accuracy'])


    #Train the model
    for epoch in range(args.epochs):
        model.train()

        total_loss = 0
        accuracy = 0
        num_batches = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data, supernode_indices, device)
            loss = loss_fn(out, data.y)
            total_loss += loss.item()
            num_batches += 1
            #Add to the accuracy the number of correct binary predictions
            accuracy += 0#(out.round(decimals=0) == data.y).sum().item()/len(data.y)

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        #Find the performance on the test set
        model.eval()
        test_accuracy = 0
        test_loss = 0
        test_num_batches = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data, supernode_indices, device)
            test_loss += loss_fn(out, data.y).item()
            test_accuracy += 0#(out.round(decimals=0) == data.y).sum().item()/len(data.y)
            test_num_batches += 1

        print('Epoch: {:03d}, Train Loss: {:.7f}, Train Accuracy: {:.3}, Test Loss: {:.7f}, Test Accuracy: {:.3}'.format(epoch, 
               total_loss/num_batches, accuracy/num_batches, test_loss/test_num_batches, test_accuracy/test_num_batches))

        #Store the results in the dataframe
        df = pd.concat([df, pd.DataFrame({'epoch': epoch, 'loss': total_loss/num_batches, 
                                          'accuracy': accuracy/num_batches, 'test_loss': test_loss/test_num_batches,
                                          'test_accuracy': test_accuracy/test_num_batches
                                          }, index=[0])], ignore_index=True)
        #Save the dataframe to a csv file
        df.to_csv('results_allcoords.csv', index=False)

        if(epoch==0):
            best_loss = test_loss/test_num_batches
        if(test_loss/test_num_batches < best_loss):
            best_loss = test_loss/test_num_batches
            torch.save(model.state_dict(), 'model_allcoords.pt')

class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

if __name__ == '__main__':

    for args in [
            {
            'time_window_size': args.time_window_size,
            'proc_nlp_hidden_dim': args.proc_nlp_hidden_dim,
            'time_nlp_hidden_dim': args.time_nlp_hidden_dim,
            'num_supernodes': args.num_supernodes,
            'super_nlp_hidden_dim_1': args.super_nlp_hidden_dim_1,
            'super_nlp_hidden_dim_2': args.super_nlp_hidden_dim_2,
            'num_layers': args.num_layers,
            'batch_size':args.batch_size,
            'epochs': args.epochs,
            'opt': args.opt,
            'opt_scheduler': args.opt_scheduler,
            'lr': args.lr,
            'device': args.device,
            'seed': args.seed,
            'weight_decay': args.weight_decay,
            'train_ratio': args.train_ratio
            },
        ]:
            args = objectview(args)

    dataset = torch.load('/workspace/data_gen/pupil_allcoords_graphs.pt')

    first_graph = dataset[0]

    #Find the supernode indices
    random_indices = np.random.choice(first_graph.edge_index.shape[-1], size=args.num_supernodes, replace=False)
    supernode_indices = first_graph.edge_index[0,random_indices]

    #remove duplicate nodes
    supernode_indices = np.unique(supernode_indices.cpu().numpy())

    final_num_supernodes = len(supernode_indices)

    print("Number of supernodes: %d" % final_num_supernodes)

    #Save the supernode indices
    np.save('supernode_indices_allcoords.npy', supernode_indices)

    #dataset.y = torch.tensor([0], dtype=torch.float).unsqueeze(-1)
    # for i, data in enumerate(dataset):
    #     print(data.y.item())
    #     # if(np.isclose(data.y.item(), 0.0)):
    #     #     print("Index: %d" % i)
    # import pdb;pdb.set_trace()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    print("device in use: {}".format(device))




    #Train the model
    train(dataset, supernode_indices, device, args)