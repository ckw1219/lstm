import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import scipy.io as scio
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,TensorDataset


def load_data(data_path):
    
    datafile = data_path

    data = scio.loadmat(datafile)

    #4*360*100
    data_new = np.array(data['data'])
    c = data_new[:,:,0]
    for i in range(1,100):
        b = data_new[:,:,i]
        c = np.hstack((c,b))

    data_xy = c[0:2,:]
    data_v_theta = c[2:4,:]

    return data_xy,data_v_theta

datafile = 'F:\研究生\无人机项目\二维条件下计算轨迹和时间//data_t_10.mat'

data_xy,data_v_theta = load_data(datafile)

data_xy_t = data_xy.T#转置数据，打乱数据默认维度为1
data_v_theta_t = data_v_theta.T

np.random.shuffle(data_xy_t)#随机打乱数据
np.random.shuffle(data_v_theta_t)

data_xy_torch = torch.from_numpy(data_xy_t)
data_v_theta_torch = torch.from_numpy(data_v_theta_t)


train_xy = data_xy_torch[0:20000,:]
test_xy = data_xy_torch[20000:36000,:]

train_v_theta = data_v_theta_torch[0:20000,:]
test_v_theta = data_v_theta_torch[20000:36000,:]

train_xy_3 = train_xy.unsqueeze(2)
test_xy_3 = test_xy.unsqueeze(2)

train_v_theta_3 = train_v_theta.unsqueeze(2)
test_v_theta_3 = test_v_theta.unsqueeze(2)

train_dataset = TensorDataset(train_xy_3,train_v_theta_3)

train_loader = DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)
# for epoch in range(2):
#     for i,data in enumerate(train_loader):
#         inputs,label = data
#         print(inputs.data.size(),label.data.size())
class RNN(nn.Module):
    def __init__(self,input_size):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(input_size= input_size,hidden_size=64,num_layers=1,batch_first=True)
        self.out = nn.Sequential(nn.Linear(64,2))

    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out)
        return out 

Lr = 0.1
epoch = 60
rnn = RNN(2)
rnn = rnn.double()
optimizer = optim.Adam(rnn.parameters(),lr=Lr)

loss_func = nn.MSELoss()

for step in range(epoch):
    for i,data in enumerate(train_loader):
        train_input,train_label = data
        output = rnn(train_input.transpose(1,2))
        loss = loss_func(output,train_label.transpose(1,2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(step,loss)




    
