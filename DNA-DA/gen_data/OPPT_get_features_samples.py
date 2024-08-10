import numpy as np
from read_dataset import read_OPPT_dataset

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# OPPT_dataset
sensor_channels_required = ['IMU_BACK_ACC_X', 'IMU_BACK_ACC_Y', 'IMU_BACK_ACC_Z',
                            'IMU_BACK_GYRO_X', 'IMU_BACK_GYRO_Y', 'IMU_BACK_GYRO_Z',  # back
                            'IMU_RLA_ACC_X', 'IMU_RLA_ACC_Y', 'IMU_RLA_ACC_Z',
                            'IMU_RLA_GYRO_X', 'IMU_RLA_GYRO_Y', 'IMU_RLA_GYRO_Z']  # right lower arm

sensor_channels_required = ['IMU_RLA_ACC_X', 'IMU_RLA_ACC_Y', 'IMU_RLA_ACC_Z',
                            'IMU_RLA_GYRO_X', 'IMU_RLA_GYRO_Y', 'IMU_RLA_GYRO_Z',
                            'IMU_RLA_MAG_X', 'IMU_RLA_MAG_Y', 'IMU_RLA_MAG_Z']  # right lower arm

activity_list = ['Stand', 'Walk', 'Sit', 'Lie']

# activity_list = ['Open Door 1', 'Open Door 2', 'Close Door 1', 'Close Door 2',
#                 'Open Fridge', 'Close Fridge', 'Open Dishwasher', 'Close Dishwasher',
#                 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2', 'Close Drawer 2',
#                 'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup',
#                 'Toggle Switch']


activities_required = activity_list
source_user = 'S1'
target_user = 'S3'  # S3
Sampling_frequency = 30  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5
DATASET_NAME = 'OPPT'
oppt_ds = read_OPPT_dataset.READ_OPPT_DATASET(source_user, target_user, bag_window_second=Num_Seconds,
                                              bag_overlap_rate=Window_Overlap_Rate,
                                              instances_window_second=0.1, instances_overlap_rate=0.5,
                                              sampling_frequency=Sampling_frequency)

source_required_X_bags, source_required_Y_bags, source_required_amount, _, _, \
target_required_X_bags, target_required_Y_bags, target_required_amount, _, _, _, _, _, _ \
    = oppt_ds.generate_data_with_required_sensor_channels_and_activities(sensor_channels_required, activities_required)

source_required_X_bags = np.array(source_required_X_bags)
source_required_Y_bags = np.array(source_required_Y_bags)-1
target_required_X_bags = np.array(target_required_X_bags)
target_required_Y_bags = np.array(target_required_Y_bags)-1
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# data persistent
DATASET_NAME = "OPPT"
with open(DATASET_NAME + '_all_' + str(source_user) + '_X.npy', 'wb') as f:
    np.save(f, source_required_X_bags)
with open(DATASET_NAME + '_all_' + str(source_user) + '_Y.npy', 'wb') as f:
    np.save(f, source_required_Y_bags)
with open(DATASET_NAME + '_all_' + str(target_user) + '_X.npy', 'wb') as f:
    np.save(f, target_required_X_bags)
with open(DATASET_NAME + '_all_' + str(target_user) + '_Y.npy', 'wb') as f:
    np.save(f, target_required_Y_bags)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
