import numpy as np
from read_dataset import read_PAMAP2_dataset

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# PAMAP2_dataset
source_user = '7'  # 1 # 2 # 5 # 6 # 8
target_user = '1'
Sampling_frequency = 100  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5
Sensor_P = "H" # "HCA"

pamap2_ds = read_PAMAP2_dataset.READ_PAMAP2_DATASET(source_user, target_user, window_second=Num_Seconds,
                                                    window_overlap_rate=Window_Overlap_Rate,
                                                    sampling_frequency=Sampling_frequency,
                                                    Sensor_P=Sensor_P)

source_windows, source_labels, source_timestamps, target_windows, target_labels, target_timestamps = pamap2_ds.filter_and_sort_data()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# data persistent
DATASET_NAME = "PAMAP2"
with open(DATASET_NAME + '_all_' + str(source_user) + '_X.npy', 'wb') as f:
    np.save(f, np.array(source_windows, dtype=object))
with open(DATASET_NAME + '_all_' + str(source_user) + '_Y.npy', 'wb') as f:
    np.save(f, np.array(source_labels))
with open(DATASET_NAME + '_all_' + str(target_user) + '_X.npy', 'wb') as f:
    np.save(f, np.array(target_windows, dtype=object))
with open(DATASET_NAME + '_all_' + str(target_user) + '_Y.npy', 'wb') as f:
    np.save(f, np.array(target_labels))
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
