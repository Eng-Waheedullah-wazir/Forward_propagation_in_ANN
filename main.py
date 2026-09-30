import torch 
import torch.nn as nn
import pandas as pd
from dataset import  Dataset
from DataLoader import create_data
from train_model import Train_model
# read data from the csv.
data="data_path"
dataset=Dataset(data)
x_train,y_train,x_val,y_val,x_test,y_test=dataset.split_data()
train_data,val_data,test_data=create_data(x_train,y_train,x_val,y_val,x_test,y_test)
# create  the object of the Trainmodel
model=Train_model(train_data.shape[1])
out=model(train_data)
print(out.shape)

