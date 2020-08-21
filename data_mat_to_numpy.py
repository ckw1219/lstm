import scipy.io as scio
import numpy as np


'''
将matlab格式的数据mat转换成Python中可以用的数据
转换后的数据是字典格式
'''
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
