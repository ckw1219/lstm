import scipy.io as scio
import numpy as np

# datafile = 'F:\研究生\无人机项目\二维条件下计算轨迹和时间//data_t_10.mat'

# data = scio.loadmat(datafile)

# #4*360*100
# data_new = np.array(data['data'])
# c = data_new[:,:,0]
# for i in range(1,100):
#     b = data_new[:,:,i]
#     c = np.hstack((c,b))

# data_xy = c[0:2,:]
# data_v_theta = c[2:4,:]

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


path = 'F:\研究生\无人机项目\二维条件下计算轨迹和时间//data_t_10.mat'

data_xy,data_v_theta = load_data(path)
print(data_xy.shape)
print(type(data_xy))

data = []
for i in range(36000):
    data.append(data_xy[:,i])    


data_in = ()
print(len(data))
print(type(data))
print(data)


# x = []
# for i in range(4):
#     x.append(data_v_theta[:,i])

# xn = np.array(x)
# print(type(x))
# print(type(xn))
# print(data_xy[:,0],data_v_theta[:,0])
# import numpy as np

# x_dim = 2
# y_list = [-0.5, 0.2, 0.1, -0.5]
# input_val_arr = [np.random.random(x_dim) for _ in y_list]

# x = []
# for i in range(4):
#     x.append(data_v_theta[:,i]
# print(type(np.random.random(x_dim)))
# print(input_val_arr.shape)
# print(y_list)
# print(input_val_arr)

# A = np.random.random(50)
# print(A)