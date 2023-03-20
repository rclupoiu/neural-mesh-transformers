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





#Find the columns of the correlation matrix that have at least one nonzero value
nonzero_cols = np.where(corr_matrix.sum(axis=0) > 0)[0]

#Find the rows of the correlation matrix that have at least one nonzero value
nonzero_rows = np.where(corr_matrix.sum(axis=1) > 0)[0]

#Filter the neuron data to only include the neurons that have at least one nonzero correlation
compressed_neuron_data = neuron_data[nonzero_cols]

#Convert the correlation matrix to remove the rows and columns that are all zeros
compressed_corr_matrix = corr_matrix[nonzero_cols][:, nonzero_cols]







#Convert the correlation matrix to a list of graph edge connections in COO format
edges = np.nonzero(compressed_corr_matrix)
edge_index = torch.tensor(np.array([edges[0], edges[1]]), dtype=torch.long)

#Define the edge attributes as the correlation values between the neurons
edge_attr = torch.tensor(compressed_corr_matrix[edges], dtype=torch.float).unsqueeze(-1)




#Create a dataset of graphs and pupil direction labels
dataset_size = 6500
data_list = []

left_coord_threshold = 70
right_coord_threshold = 85

#Define the node attributes as a vector containing the neuron's data for 20 preceding frames, including the current frame
frame_window = 20

#Find the frames where the pupil is looking either to the left or right and is greater than 20 frames from the beginning of the recording
#eligible_frames = np.where((pupil_x_coords < left_coord_threshold) | (pupil_x_coords > right_coord_threshold))[0]
#eligible_frames = eligible_frames[eligible_frames > frame_window]
eligible_frames = np.arange(frame_window, len(pupil_x_coords))

print("Collecting Data...")

num_left = 0
num_right = 0

# #Filter the x coordinates to remove nan values
# pupil_x_coords = pupil_x_coords[~np.isnan(pupil_x_coords)]

# #Normalize the x coordinates to have a mean of 0 and a standard deviation of 1
# pupil_x_coords = (pupil_x_coords - np.mean(pupil_x_coords)) / np.std(pupil_x_coords)

# Remove values from x coordinates at indices where either x or y coordinates are nan
filtered_pupil_x_coords = pupil_x_coords[~np.isnan(pupil_x_coords) & ~np.isnan(pupil_y_coords)]
# Do the same for y coordinates
filtered_pupil_y_coords = pupil_y_coords[~np.isnan(pupil_x_coords) & ~np.isnan(pupil_y_coords)]

#Normalize the x coordinates to have a mean of 0 and a standard deviation of 1
filtered_pupil_x_coords = (filtered_pupil_x_coords - np.mean(filtered_pupil_x_coords)) / np.std(filtered_pupil_x_coords)

#Normalize the y coordinates to have a mean of 0 and a standard deviation of 1
filtered_pupil_y_coords = (filtered_pupil_y_coords - np.mean(filtered_pupil_y_coords)) / np.std(filtered_pupil_y_coords)

for i in range(dataset_size):

    curr_frame = eligible_frames[i]

    node_attr = torch.tensor(compressed_neuron_data[:,curr_frame-frame_window:curr_frame+1], dtype=torch.float)

    #Define the graph label as the current frame's direction of the pupil in the x direction
    #If the x coordinate of the pupil is less than left_coord_threshold, the direction is left, denoted by 0
    #If the x coordinate of the pupil is greater than right_coord_threshold, the direction is right, denoted by 1

    curr_x_coord = filtered_pupil_x_coords[curr_frame]
    curr_y_coord = filtered_pupil_y_coords[curr_frame]

    graph_label = torch.tensor([curr_x_coord, curr_y_coord], dtype=torch.float).unsqueeze(-1)

    # if(curr_x_coord < left_coord_threshold):
    #     graph_label = torch.tensor([0], dtype=torch.float).unsqueeze(-1)
    #     num_left += 1
    # elif(curr_x_coord > right_coord_threshold):
    #     graph_label = torch.tensor([1], dtype=torch.float).unsqueeze(-1)
    #     num_right += 1
    # else:
    #     graph_label = torch.tensor([-1], dtype=torch.float).unsqueeze(-1)
    #     print("WARNING: Pupil located in the middle")

    data_list.append(Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=graph_label))

print("Saving Data...")

torch.save(data_list, 'pupil_allcoords_graphs.pt')

print("Done Saving Data!")
