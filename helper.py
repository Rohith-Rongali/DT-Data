import matplotlib.pyplot as plt
from dataclasses import dataclass, replace

from data import generate_points,TreeNode,gen_std_basis_DT,CustomDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch


def return_data_elements(DataConfig):
    # returns [data] and data_loader

    x_train,y_train = gen_std_basis_DT(depth = DataConfig.depth, dim_in = DataConfig.dim_in, num_points = DataConfig.num_points,type_data= DataConfig.type_data, radius = DataConfig.radius)
    x_test,y_test = gen_std_basis_DT(depth = DataConfig.depth, dim_in = DataConfig.dim_in, num_points = DataConfig.num_points,type_data= DataConfig.type_data, radius = DataConfig.radius)
    # x_train, x_test, y_train, y_test = train_test_split(
    # X, Y, test_size=0.1, random_state=42, stratify=Y)

    data = [x_train,y_train,x_test,y_test]

    return data

def loaders(data,batch_size=32):
    x_train,y_train,x_test,y_test = data
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader



    