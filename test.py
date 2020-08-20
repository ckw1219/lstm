import numpy as np

from lstm import LstmParam, LstmNetwork
import scipy.io as scio

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(100):
        print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])

        print("y_pred = [" +
              ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

def load_data(path):

    datafile = path

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

def example_1():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)
    
    data_path = 'F:\研究生\无人机项目\二维条件下计算轨迹和时间//data_t_10.mat'
    data_xy,data_v_theta = load_data(data_path)
    
    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 2
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    y_list = data_v_theta[0,:]

    #input_val_arr = [data.append(data_xy[:,i]) for i in range(36000)]
    data = []
    for i in range(36000):
        data.append(data_xy[:,i])    
    input_val_arr  = data

    for cur_iter in range(100):
        print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])

        print("y_pred = [" +
              ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

if __name__ == "__main__":

    example_1()

