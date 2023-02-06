import torch
import numpy as np

def sum_data(data_list, index):    
    if index == 0:
        return 0
    else:
        return sum(data_list[:index])

def remove_2d_outlier(data, p_outlier, gap_list ,y_center , varaiance_list, positive_limit):
    for i in range(len(data)):
        distance = 0
        for j in range(len(varaiance_list)):
            distance += 1. / (2. * np.pi * varaiance_list[j] *  varaiance_list[j]) * np.exp(-((data[i,0] - gap_list[j])**2. / (2. * varaiance_list[j] **2.) + (data[i,1] - y_center[j])**2. / (2. * varaiance_list[j] **2.)))
        while distance/len(varaiance_list) > p_outlier:  # 2
            outlier = np.random.random((2))
            data[i,:] = outlier * (2*positive_limit) - positive_limit
            distance = 0
            for j in range(len(varaiance_list)):
                distance += 1. / (2. * np.pi * varaiance_list[j] *  varaiance_list[j]) * np.exp(-((data[i,0] - gap_list[j])**2. / (2. * varaiance_list[j] **2.) + (data[i,1] - y_center[j])**2. / (2. * varaiance_list[j] **2.)))
    return data

    

def generate_data(batch_size,input_dim, class_num, gap_list,y_center, variance, train_data_num ):
    data = np.zeros((batch_size, input_dim))
    for i in range(class_num):
        mean = ( gap_list[i] ,y_center[i] )
        cov = np.diag([variance, variance])
        data[sum_data(train_data_num, i): sum_data(train_data_num, i+1),:  ] = np.random.multivariate_normal(mean, cov,  train_data_num[i])
    data  = torch.from_numpy(data).float()

    label = torch.zeros((batch_size))
    for i in range(class_num):
        label[sum_data(train_data_num, i): sum_data(train_data_num, i+1) ] = torch.ones(train_data_num[i])*i 
    label = label.long()
    return data, label
