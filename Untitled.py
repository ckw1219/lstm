import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import scipy.io as scio
import numpy as np

# def load_data(data_path):
    
#     datafile = data_path

#     data = scio.loadmat(datafile)

#     #4*360*100
#     data_new = np.array(data['data'])
#     c = data_new[:,:,0]
#     for i in range(1,100):
#         b = data_new[:,:,i]
#         c = np.hstack((c,b))

#     data_xy = c[0:2,:]
#     data_v_theta = c[2:4,:]

#     return data_xy,data_v_theta

# datafile = 'F:\研究生\无人机项目\二维条件下计算轨迹和时间//data_t_10.mat'

# data_xy,data_v_theta = load_data(datafile)

# data_xy_t = data_xy.T



# arr = np.array(range(12)).reshape(3, 4)
# print(arr)
# np.random.shuffle(arr)
# print(arr)

# rnn = nn.lstm(10,20,2)
# input = torch.randn(5,3,10)
# h0 = torch.randn(2,3,20)
# c0 = torch.randn(2,3,20)
# output,(hn.cn) = rnn(input,(h0,c0))

rnn = nn.LSTM(10,20,2)#输入向量维数10, 隐藏元维度20, 2个LSTM层串联(若不写则默认为1）
input = torch.randn(5,3,10)#输入（seq_len, batch, input_size） 序列长度为5 batch为3 输入维度为10
h0 = torch.randn(2,3,20)#h_0(num_layers * num_directions, batch, hidden_size)  num_layers = 2 ，batch=3 ，hidden_size = 20
c0 = torch.randn(2,3,20)#同上
output, (hn,cn) = rnn(input, (h0,c0))

print(output.size())

print(input.transpose(1,2).size)