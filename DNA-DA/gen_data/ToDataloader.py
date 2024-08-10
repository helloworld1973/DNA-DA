import torch
import torch.utils.data as Data
import numpy as np


def get_data_in_dataloader(S_x, S_y, T_x, T_y, batch_size, device):
    S_x = np.array(S_x, dtype=np.float32)
    T_x = np.array(T_x, dtype=np.float32)
    # Process S data
    S_x_func = torch.tensor(S_x, dtype=torch.float32).to(device).permute(0, 2, 1).unsqueeze(2)
    S_y_func = torch.tensor(S_y).to(device)
    S_d_func = torch.zeros(len(S_y_func)).to(device)  # Simplified from a loop
    S_torch_dataset = Data.TensorDataset(S_x_func, S_y_func, S_d_func)
    S_torch_loader = Data.DataLoader(dataset=S_torch_dataset, batch_size=batch_size, shuffle=False)

    # Process T data
    T_x_func = torch.tensor(T_x, dtype=torch.float32).to(device).permute(0, 2, 1).unsqueeze(2)
    T_y_func = torch.tensor(T_y).to(device)
    T_d_func = torch.ones(len(T_y_func)).to(device)  # Simplified from a loop
    T_torch_dataset = Data.TensorDataset(T_x_func, T_y_func, T_d_func)
    T_torch_loader = Data.DataLoader(dataset=T_torch_dataset, batch_size=batch_size, shuffle=False)

    # Process combined ST data
    num_class = np.unique(S_y).shape[0]
    S_index_array = np.arange(len(S_y_func))
    S_index_array = torch.from_numpy(S_index_array).to(device)

    ST_x = np.concatenate((S_x, T_x))
    ST_x_func = torch.tensor(ST_x, dtype=torch.float32).to(device).permute(0, 2, 1).unsqueeze(2)
    T_y_new = np.array([i + num_class for i in T_y])
    ST_y = np.concatenate((S_y, T_y_new))
    ST_y_func = torch.tensor(ST_y).to(device)
    ST_d_func = torch.cat([S_d_func, T_d_func]).to(device)
    ST_torch_dataset = Data.TensorDataset(ST_x_func, ST_y_func, ST_d_func)
    ST_torch_loader = Data.DataLoader(dataset=ST_torch_dataset, batch_size=batch_size, shuffle=False)

    return S_torch_loader, T_torch_loader, ST_torch_loader, S_index_array
