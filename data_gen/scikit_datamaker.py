import numpy as np
import scipy

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

#Go through each of the neuron time traces and normalize them to have a mean of 0 and a standard deviation of 1
for i in range(len(compressed_neuron_data)):
    compressed_neuron_data[i] = (compressed_neuron_data[i] - np.mean(compressed_neuron_data[i])) / np.std(compressed_neuron_data[i])


dataset_size = 6500

#Define the node attributes as a vector containing the neuron's data for 20 preceding frames, including the current frame
frame_window = 20

eligible_frames = np.arange(frame_window, len(pupil_x_coords))

print("Collecting Data...")


# Remove values from x coordinates at indices where either x or y coordinates are nan
filtered_pupil_x_coords = pupil_x_coords[~np.isnan(pupil_x_coords) & ~np.isnan(pupil_y_coords)]
# Do the same for y coordinates
filtered_pupil_y_coords = pupil_y_coords[~np.isnan(pupil_x_coords) & ~np.isnan(pupil_y_coords)]

#Normalize the x coordinates to have a mean of 0 and a standard deviation of 1
filtered_pupil_x_coords = (filtered_pupil_x_coords - np.mean(filtered_pupil_x_coords)) / np.std(filtered_pupil_x_coords)

#Normalize the y coordinates to have a mean of 0 and a standard deviation of 1
filtered_pupil_y_coords = (filtered_pupil_y_coords - np.mean(filtered_pupil_y_coords)) / np.std(filtered_pupil_y_coords)

X = np.zeros((dataset_size, len(compressed_neuron_data), frame_window+1))
y = np.zeros((dataset_size, 2))

print("Collecting Data...")

for i in range(dataset_size):

    curr_frame = eligible_frames[i]

    X[i] = compressed_neuron_data[:, curr_frame - frame_window:curr_frame+1]

    y[i] = np.array([filtered_pupil_x_coords[curr_frame], filtered_pupil_y_coords[curr_frame]])

print("Data Collected")

np.save('X.npy', X)
np.save('y.npy', y)

print("Data Saved")