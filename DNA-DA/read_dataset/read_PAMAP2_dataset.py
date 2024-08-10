import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
from sklearn.preprocessing import StandardScaler

# sampling frequency: 100Hz
class READ_PAMAP2_DATASET:
    def __init__(self, source_user, target_user, window_second, window_overlap_rate, sampling_frequency, Sensor_P):
        self.File_Path = 'C:\\Users\\yxzhj\\PycharmProjects\\TL_DATASETS_REPO\\PAMAP2_Dataset\\Protocol\\'
        self.source_user = source_user
        self.target_user = target_user
        self.window_second = window_second
        self.window_overlap_rate = window_overlap_rate
        self.sampling_frequency = sampling_frequency

        self.S_file_path = self.File_Path + 'subject10' + self.source_user + '.dat'
        self.T_file_path = self.File_Path + 'subject10' + self.target_user + '.dat'

        self.IMU_Hand = ['handAcc16_1', 'handAcc16_2', 'handAcc16_3',
                         'handGyro1', 'handGyro2', 'handGyro3',
                         'handMagne1', 'handMagne2', 'handMagne3']
        self.IMU_Chest = ['chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3',
                          'chestGyro1', 'chestGyro2', 'chestGyro3',
                          'chestMagne1', 'chestMagne2', 'chestMagne3']
        self.IMU_Ankle = ['ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
                          'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                          'ankleMagne1', 'ankleMagne2', 'ankleMagne3']
        self.Col_Names = ["timestamp", "activityID"]

        self.Column_Names = (["timestamp", "activityID", "heartrate"] +
                             ['handTemperature',
                              'handAcc16_1', 'handAcc16_2', 'handAcc16_3',
                              'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
                              'handGyro1', 'handGyro2', 'handGyro3',
                              'handMagne1', 'handMagne2', 'handMagne3',
                              'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4'] +
                             ['chestTemperature',
                              'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3',
                              'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
                              'chestGyro1', 'chestGyro2', 'chestGyro3',
                              'chestMagne1', 'chestMagne2', 'chestMagne3',
                              'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4'] +
                             ['ankleTemperature',
                              'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
                              'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3',
                              'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                              'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
                              'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'])

        self.Activity_Mapping = {'transient': 0,
                                 'lying': 1,
                                 'sitting': 2,
                                 'standing': 3,
                                 'walking': 4,
                                 'running': 5,
                                 'cycling': 6,
                                 'Nordic_walking': 7,
                                 'watching_TV': 9,
                                 'computer_work': 10,
                                 'car_driving': 11,
                                 'ascending_stairs': 12,
                                 'descending_stairs': 13,
                                 'vacuum_cleaning': 16,
                                 'ironing': 17,
                                 'folding_laundry': 18,
                                 'house_cleaning': 19,
                                 'playing_soccer': 20,
                                 'rope_jumping': 24}

        if Sensor_P == "H":
            self.Sensor_Dict = [self.IMU_Hand]
        elif Sensor_P == "C":
            self.Sensor_Dict = [self.IMU_Chest]
        elif Sensor_P == "A":
            self.Sensor_Dict = [self.IMU_Ankle]
        elif Sensor_P == "HC":
            self.Sensor_Dict = [self.IMU_Hand, self.IMU_Chest]
        elif Sensor_P == "HA":
            self.Sensor_Dict = [self.IMU_Hand, self.IMU_Ankle]
        elif Sensor_P == "CA":
            self.Sensor_Dict = [self.IMU_Chest, self.IMU_Ankle]
        elif Sensor_P == "HCA":
            self.Sensor_Dict = [self.IMU_Hand, self.IMU_Chest, self.IMU_Ankle]

    def read_data_as_df_from_file_path(self, dat_file_path):
        # Read the .DAT file using np.loadtxt
        data = np.loadtxt(dat_file_path)

        # Convert the data to a DataFrame
        df = pd.DataFrame(data, columns=self.Column_Names)

        print(df)

        columns = self.Col_Names.copy()
        print(columns)
        for a_position_sensor in self.Sensor_Dict:
            columns += a_position_sensor
        print(columns)
        df = df.loc[:, columns]
        df = df.dropna()

        return df

    def get_common_activities_and_reindex(self, S_df, T_df, activity_mapping):
        # Find common activity IDs
        common_activity_ids = set(S_df['activityID']).intersection(set(T_df['activityID']))

        # Create a new mapping for common activities
        common_activities = {activity: activity_mapping[activity] for activity in activity_mapping
                             if activity_mapping[activity] in common_activity_ids}

        # Reindex the common activities
        new_index = {original: new - 1 for new, (activity, original) in enumerate(common_activities.items())}

        # Create the new table
        new_table = []
        for activity, original_index in activity_mapping.items():
            new_table.append({
                "Activity": activity,
                "Original Index": original_index,
                "New Order Index": new_index.get(original_index, None)
            })

        new_table_df = pd.DataFrame(new_table)

        return new_table_df, new_index

    def reindex_and_sort_data(self, df, new_index):
        # Remove data with label 0 in original index
        df = df[df['activityID'] != 0]

        # Filter data for common activities and sort by timestamp
        common_activity_ids = new_index.keys()
        filtered_df = df[df['activityID'].isin(common_activity_ids)].sort_values(by='timestamp')

        # Replace activityID with new indices
        filtered_df['activityID'] = filtered_df['activityID'].map(new_index)

        return filtered_df

    def apply_sliding_window(self, one_df):
        data = one_df.iloc[:, 2:].values

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        # print(source_data_combined.shape)

        labels = one_df.iloc[:, 1].values.reshape(-1, 1)
        # print(source_labels_combined.shape)

        timestamp = one_df.iloc[:, 0].values.reshape(-1, 1)
        # print(source_timestamp_combined.shape)

        df = pd.DataFrame(data)
        df['label'] = labels
        df['timestamp'] = timestamp

        # Initialize lists to hold the resulting windows, labels, and timestamps
        all_windows = []
        all_labels = []
        all_timestamps = []

        # Group the data by activity label
        grouped = df.groupby('label')

        # Iterate over each group
        for label, group in grouped:
            # Sort the group by timestamp
            group = group.sort_values(by='timestamp')

            window_size = self.window_second * self.sampling_frequency
            stride = int(window_size * self.window_overlap_rate)

            # Apply the sliding window to the sorted group
            data_windows = SlidingWindow(size=window_size, stride=stride).fit_transform(
                group.iloc[:, :-2].values)  # Exclude the label and timestamp columns

            # Extract labels and timestamps for each window
            for i in range(data_windows.shape[0]):
                start_idx = i * stride
                end_idx = start_idx + window_size
                window_labels = group['label'].values[start_idx:end_idx]
                window_timestamps = group['timestamp'].values[start_idx:end_idx]

                # Use the last label and the range of timestamps for the current window
                all_windows.append(data_windows[i])
                all_labels.append(window_labels[-1])
                all_timestamps.append(window_timestamps)

        return np.array(all_windows), np.array(all_labels), np.array(all_timestamps)

    def filter_and_sort_data(self):

        S_df_raw_data_given_positions = self.read_data_as_df_from_file_path(self.S_file_path)
        T_df_raw_data_given_positions = self.read_data_as_df_from_file_path(self.T_file_path)

        # Get the new table with reindexed activities and new index mapping
        new_table_df, new_index = self.get_common_activities_and_reindex(S_df_raw_data_given_positions,
                                                                         T_df_raw_data_given_positions,
                                                                         self.Activity_Mapping)

        # Reindex and sort data separately for S and T
        S_reindexed_sorted_df = self.reindex_and_sort_data(S_df_raw_data_given_positions, new_index)
        T_reindexed_sorted_df = self.reindex_and_sort_data(T_df_raw_data_given_positions, new_index)

        source_windows, source_labels, source_timestamps = self.apply_sliding_window(S_reindexed_sorted_df)
        target_windows, target_labels, target_timestamps = self.apply_sliding_window(T_reindexed_sorted_df)

        return source_windows, source_labels, source_timestamps, target_windows, target_labels, target_timestamps
