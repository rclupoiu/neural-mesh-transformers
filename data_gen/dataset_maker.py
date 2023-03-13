import numpy as np
import matplotlib.pyplot as plt
import scipy

import torch
from torch_geometric.data import Data
import networkx as nx

dat = np.load('stringer_spontaneous.npy', allow_pickle=True).item()
neuron_data = dat['sresp']

pupil_x_coords = dat['pupilCOM'][:,0]
pupil_y_coords = dat['pupilCOM'][:,1]

#Filter the neuron data with a backwards-looking window of 20 frames
frame_size = 20
neuron_data_filtered = scipy.ndimage.uniform_filter1d(neuron_data, frame_size, axis=-1, mode='nearest', origin=-frame_size//2)

corr_matrix = np.corrcoef(neuron_data_filtered).round(decimals=4)

#Find the largest 2%, positive, non-diagonal correlations
corr_matrix[np.diag_indices_from(corr_matrix)] = 0
corr_matrix[corr_matrix < 0] = 0

# NOTE: Since the correlation matrix is symmetric, we must take the top 10% of the values
# in the entire matrix to get the top 10% of the connections
corr_matrix[corr_matrix < np.percentile(corr_matrix, 99)] = 0



#Convert the correlation matrix to a list of graph edge connections in COO format
edges = np.nonzero(corr_matrix)
edge_index = torch.tensor(np.array([edges[0], edges[1]]), dtype=torch.long)

#Define the edge attributes as the correlation values between the neurons
edge_attr = torch.tensor(corr_matrix[edges], dtype=torch.float).unsqueeze(-1)




#Create a dataset of graphs and pupil direction labels
dataset_size = 1000
data_list = []

print("Collecting Data...")

for i in range(dataset_size):
    #Define the node attributes as a vector containing the neuron's data for 20 preceding frames, including the current frame
    frame_window = 20

    #Find the frames where the pupil is looking either to the left or right and is greater than 20 frames from the beginning of the recording
    eligible_frames = np.where((pupil_x_coords < 75) | (pupil_x_coords > 85))[0]
    eligible_frames = eligible_frames[eligible_frames > frame_window]

    curr_frame = eligible_frames[i]

    #Extract the neuron indices from the edge connections
    neuron_indices = np.unique(edges[0])

    complete_neuron_indices = np.arange(neuron_indices.max())



    #Find the values in complete_neuron_indices that are not in neuron_indices
    #missing_neuron_indices = np.setdiff1d(complete_neuron_indices, neuron_indices)

    #Construct the node attributes to contain nan values for the missing neurons
    node_attr = torch.empty((complete_neuron_indices.shape[0]+1, frame_window+1), dtype=torch.float)
    node_attr[:] = torch.nan
    node_attr[neuron_indices] = torch.tensor(neuron_data[neuron_indices,curr_frame-frame_window:curr_frame+1], dtype=torch.float)

    #Check if there are any nan in the location where there should be a neuron
    if(torch.isnan(node_attr[neuron_indices]).any()):
        print("WARNING: Indexing error; message passing will include nan values")

    if(torch.isnan(node_attr[edge_index[0]]).any()):
        print("WARNING: Indexing error; message passing will include nan values")

    if(torch.isnan(node_attr[edge_index[1]]).any()):
        print("WARNING: Indexing error; message passing will include nan values")




    #Define the graph label as the current frame's direction of the pupil in the x direction
    #If the x coordinate of the pupil is less than 75, the direction is left, denoted by 0
    #If the x coordinate of the pupil is greater than 85, the direction is right, denoted by 1

    curr_x_coord = pupil_x_coords[curr_frame]
    if(curr_x_coord < 75):
        graph_label = torch.tensor([0], dtype=torch.float).unsqueeze(-1)
    elif(curr_x_coord > 85):
        graph_label = torch.tensor([1], dtype=torch.float).unsqueeze(-1)
    else:
        graph_label = torch.tensor([-1], dtype=torch.float).unsqueeze(-1)
        print("WARNING: Pupil located in the middle")

    data_list.append(Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=graph_label))

print("Saving Data...")

torch.save(data_list, 'pupil_direction_graphs.pt')

print("Done Saving Data!")
