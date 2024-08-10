import random
import math
import numpy as np
import torch

from gen_data.ToDataloader import get_data_in_dataloader
from gen_model.alg.Diffusion_network import DiffusionModel

from gen_model.alg.model_validation import get_accuracy_user
from gen_model.utils.util import log_and_print, log_confusion_matrix

########################################################################################################################
# set seed
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
########################################################################################################################

########################################################################################################################
# read OPPT dataset
source_user = 'S1'
target_user = 'S2'  # S3
Sampling_frequency = 30  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5
DATASET_NAME = 'OPPT'

with open('./gen_data/' + DATASET_NAME + '_all_' + str(source_user) + '_X.npy', 'rb') as f:
    source_windows = np.load(f, allow_pickle=True)
with open('./gen_data/' + DATASET_NAME + '_all_' + str(source_user) + '_Y.npy', 'rb') as f:
    source_labels = np.load(f)
with open('./gen_data/' + DATASET_NAME + '_all_' + str(target_user) + '_X.npy', 'rb') as f:
    target_windows = np.load(f, allow_pickle=True)
with open('./gen_data/' + DATASET_NAME + '_all_' + str(target_user) + '_Y.npy', 'rb') as f:
    target_labels = np.load(f)
########################################################################################################################

########################################################################################################################
# log file name setting
file_name = str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_DiffNoiseAdvDA.txt'
file_name_summary = str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_DiffNoiseAdvDA_summary.txt'
########################################################################################################################

########################################################################################################################
# transfer to dataloader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 5000
S_torch_loader, T_torch_loader, ST_torch_loader, S_x_index = get_data_in_dataloader(source_windows, source_labels,
                                                                                    target_windows, target_labels,
                                                                                    batch_size, device)
########################################################################################################################

########################################################################################################################
# model training
global_epoch = 500
num_class = len(np.unique(source_labels))
conv1_in_channels = 9  # fixed # 6
Alpha = 1.0  # RECON_L
Beta = 1.0  # NOISE_CLASS_L
Delta = 1.0  # DENOISE_CLASS_R_L
LR = 0.001  # learning rate

# n_t = 40
# beta_1 = 0.0004
# beta_scale_rate = 50
# beta_t = beta_scale_rate * beta_1
# conv1_out_channels = 64
# conv2_out_channels = 32  # 128
# conv_kernel_size_num = 15
# pool_kernel_size_num = 3
# in_features_size = conv2_out_channels * math.floor(((Num_Seconds * Sampling_frequency - conv_kernel_size_num + 1) / pool_kernel_size_num - conv_kernel_size_num + 1) / pool_kernel_size_num)
# denoise_hidden_size = 64
# discriminator_dis_hidden = 256
# ReverseLayer_latent_domain_alpha = 0.1


for ReverseLayer_latent_domain_alpha in [0.005, 0.01, 0.03, 0.04, 0.06, 0.5, 0.6]: # [0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4]
    for n_t in [10, 20, 30, 40, 50, 60, 70]:
        for beta_1 in [0.0001, 0.0002, 0.0004, 0.0008, 0.0010, 0.0050, 0.010]:
            for beta_scale_rate in [10, 30, 50, 70, 100]:
                for conv1_out_channels in [16, 32, 64, 128]:
                    for conv2_out_channels in [16, 32, 64, 128]:
                        for conv_kernel_size_num in [2, 5, 10, 15, 20, 30]:
                            for pool_kernel_size_num in [2, 3, 4, 5]:
                                for denoise_hidden_size in [16, 32, 64, 128]:
                                    for discriminator_dis_hidden in [16, 32, 64, 128, 256]:
                                        print('######################################################################')
                                        print('para_setting:' + str(ReverseLayer_latent_domain_alpha) + '_' + str(
                                            n_t) + '_' + str(beta_1) + '_' + str(
                                            beta_scale_rate) + '_' + str(conv1_out_channels) + '_' + str(
                                            conv2_out_channels) + '_' + str(conv_kernel_size_num) + '_' + str(
                                            pool_kernel_size_num) + '_' + str(denoise_hidden_size) + '_' + str(
                                            discriminator_dis_hidden))

                                        log_and_print(
                                            content='para_setting:' + str(ReverseLayer_latent_domain_alpha) + '_' + str(
                                                n_t) + '_' + str(beta_1) + '_' + str(
                                                beta_scale_rate) + '_' + str(conv1_out_channels) + '_' + str(
                                                conv2_out_channels) + '_' + str(conv_kernel_size_num) + '_' + str(
                                                pool_kernel_size_num) + '_' + str(denoise_hidden_size) + '_' + str(
                                                discriminator_dis_hidden), filename=file_name)

                                        in_features_size = conv2_out_channels * math.floor(((
                                                                                                    Num_Seconds * Sampling_frequency - conv_kernel_size_num + 1) / pool_kernel_size_num - conv_kernel_size_num + 1) / pool_kernel_size_num)
                                        beta_t = beta_scale_rate * beta_1
                                        algorithm = DiffusionModel(n_t=n_t, beta_1=beta_1, beta_t=beta_t, device=device,
                                                                   conv1_in_channels=conv1_in_channels,
                                                                   conv1_out_channels=conv1_out_channels,
                                                                   conv2_out_channels=conv2_out_channels,
                                                                   conv_kernel_size_num=conv_kernel_size_num,
                                                                   pool_kernel_size_num=pool_kernel_size_num,
                                                                   denoise_hidden_size=denoise_hidden_size,
                                                                   in_features_size=in_features_size,
                                                                   num_class=num_class,
                                                                   discriminator_dis_hidden=discriminator_dis_hidden,
                                                                   ReverseLayer_latent_domain_alpha=ReverseLayer_latent_domain_alpha,
                                                                   Alpha=Alpha, Beta=Beta, Delta=Delta)
                                        optimizer = torch.optim.Adam(algorithm.parameters(), lr=LR)

                                        best_S_acc = 0
                                        best_T_acc = 0
                                        which_round = 0
                                        best_T_cm = 0

                                        for round in range(global_epoch):
                                            print(str(round) + ": ----------------")
                                            log_and_print(str(round) + ": ------------------------", filename=file_name)
                                            for ST_data in ST_torch_loader:
                                                loss_result_dict = algorithm.update(ST_data, S_x_index, optimizer,
                                                                                    round)
                                                print(loss_result_dict)
                                                log_and_print(str(loss_result_dict), filename=file_name)

                                                S_acc, S_conf_matrix = get_accuracy_user(algorithm, S_torch_loader, num_class)
                                                print("S_acc: ", S_acc)
                                                log_and_print("S_acc: " + str(S_acc), filename=file_name)
                                                log_confusion_matrix(S_conf_matrix, file_name)

                                                T_acc, T_conf_matrix = get_accuracy_user(algorithm, T_torch_loader, num_class)
                                                print("T_acc: ", T_acc)
                                                log_and_print("T_acc:" + str(T_acc), filename=file_name)
                                                log_confusion_matrix(T_conf_matrix, file_name)

                                                if T_acc > best_T_acc:
                                                    best_T_acc = T_acc
                                                    best_S_acc = S_acc
                                                    which_round = round
                                                    best_T_cm = T_conf_matrix

                                        print('######################################################################')

                                        log_and_print('######################################################################', filename=file_name_summary)
                                        log_and_print('para_setting:' + str(ReverseLayer_latent_domain_alpha) + '_' + str(
                                                n_t) + '_' + str(beta_1) + '_' + str(
                                                beta_scale_rate) + '_' + str(conv1_out_channels) + '_' + str(
                                                conv2_out_channels) + '_' + str(conv_kernel_size_num) + '_' + str(
                                                pool_kernel_size_num) + '_' + str(denoise_hidden_size) + '_' + str(
                                                discriminator_dis_hidden), filename=file_name_summary)
                                        log_and_print("T_acc_best: " + str(best_T_acc), filename=file_name_summary)
                                        log_confusion_matrix(best_T_cm, file_name_summary)
                                        log_and_print("S_acc_best: " + str(best_S_acc), filename=file_name_summary)
                                        log_and_print("Round_best: " + str(which_round), filename=file_name_summary)


########################################################################################################################
