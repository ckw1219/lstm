import numpy as np
import math


'''
单隐藏层LSTM,不调用框架，直接实现
'''
#激活函数

class SigmoidActivator(object):
    def forward(self,weighted_input):
        return 1.0/(1+np.exp(-weighted_input))

    def backward(self,output): #sigmoid 的导数
        return output*(1-output)

class TanhActivator(object):#tanh的导数
    def forward(self,weighted_input):
        return 2.0/(1.0+np.exp(-2*weighted_input))-1.0

    def backward(self,output):
        return 1-output*output


#lstm的结构
class LstmLayer(object):
    def __init__(self,input_width,state_width,learning_rate):
        
        self.input_width = input_width
        self.state_width = state_width
        self.learning_rate = learning_rate

        self.gate_activator = SigmoidActivator()
        self.output_activator = TanhActivator()
        self.times = 0

        #各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        #各个时刻的单元状态向量c
        self.h_list = self.init_state_vec()
        #各个时刻的遗忘门f
        self.f_list = self.init_state_vec()
        #各个时刻的输入门i
        self.i_list = self.init_state_vec()
        #各个时刻的输出门o
        self.o_list = self.init_state_vec()
        #各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()
        #遗忘门权重矩阵wfh，wfx，偏置项bf
        self.wfh,self.wfx,self.bf = (self.init_weight_mat())
        #输入门权重
        self.wih,self.wix,self.bi = (self.init_weight_mat())
        #输出门权重
        self.woh,self.wox,self.bo = (self.init_weight_mat())
        #单元状态权重矩阵
        self.wch,self.wcx,self.bc = (self.init_weight_mat())

    def init_state_vec(self):
        '''
        初始化保存状态的向量
        '''
        state_vec_list = []
        state_vec_list.append(np.zeros((self.state_width,1)))
        return state_vec_list
    
    def init_weight_mat(self):
        '''
        初始化权重矩阵
        '''
        wh = np.random.uniform(-1e-4,1e-4,(self.state_width,self.state_width))
        wx = np.random.uniform(-1e-4,1e-4,(self.state_width,self.input_width))
        b = np.zeros((self.state_width,1))
        return wh,wx,b

    
    def forward(self,x):
        self.times += 1
        #遗忘门
        fg = self.calc_gate(x,self.wfx,self.wfh,self.bf,self.gate_activator)
        self.f_list.append(fg)
        #输入门
        ig = self.calc_gate(x,self.wix,self.wih,self.bi,self.gate_activator)
        self.i_list.append(ig)
        #输出门
        og = self.calc_gate(x,self.wox,self.woh,self.bo,self.gate_activator)
        self.o_list.append(og)
        #即时状态
        ct = self.calc_gate(x,self.wcx,self.wch,self.bc,self.output_activator)
        self.ct_list.append(ct)

        c = fg*self.c_list[self.times-1]+ig*ct
        self.c_list.append(c)

        h = og*self.output_activator.forward(c)
        self.h_list.append(h)


    def calc_gate(self,x,wx,wh,b,activator):
        h = self.h_list[self.times-1] #上次的lstm的输出
        net = np.dot(wh,h)+np.dot(wx,x)+b
        gate = activator.forward(net)
        return gate

    def backward(self,x,delta_h,activator):
        '''
        实现lstm的训练算法
        '''
        self.calc_delta(delta_h,activator)
        self.calc_gradient(x)

    def calc_delta(self,delta_h,activator):
        
        self.delta_h_list = self.init_delta()
        self.delta_o_list = self.init_delta()
        self.delta_i_list = self.init_delta()
        self.delta_f_list = self.init_delta()
        self.delta_ct_list = self.init_delta()
        #保存上一层传递下来的当前时刻的误差项
        self.delta_h_list[-1] = delta_h

        for k in range(self.times,0,-1):
            self.calc_delta_k(k)

    def init_delta(self):
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros((self.state_width,1)))
        return delta_list
    
    def calc_delta_k(self,k):
        '''
        根据k时刻的delta_h，计算k时刻的delta_f、
        delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
        '''
        ig = self.i_list[k]
        og = self.o_list[k]
        fg = self.f_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_prev = self.c_list[k-1]
        tanh_c = self.output_activator.forward(c)
        delta_k = self.delta_h_list[k]

        delta_o = (delta_k*tanh_c*self.gate_activator.backward(og))
        delta_f = (delta_k*og*(1-tanh_c*tanh_c)*c_prev*self.gate_activator.backward(fg))
        delta_i = (delta_k*og*(1-tanh_c*tanh_c)*ct*self.gate_activator.backward(ig))
        delta_ct = (delta_k*og*(1-tanh_c*tanh_c)*ig*self.output_activator.backward(ct))
        delta_h_prev = (
                np.dot(delta_o.transpose(),self.woh)+
                np.dot(delta_i.transpose(),self.wih)+
                np.dot(delta_f.transpose(),self.wfh)+
                np.dot(delta_ct.transpose(),self.wch)
        ).transpose()

        #保存全部的delta值
        self.delta_h_list[k-1] = delta_h_prev
        self.delta_f_list[k] = delta_f
        self.delta_i_list[k] = delta_i
        self.delta_o_list[k] = delta_o
        self.delta_ct_list[k] = delta_ct

    def calc_gradient(self,x):
        #初始化去找你梯度矩阵和偏置项
        self.wfh_grad,self.wfx_grad,self.bf_grad = (self.init_weight_gradient_mat())
        self.wih_grad,self.wix_grad,self.bi_grad = (self.init_weight_gradient_mat())
        self.woh_grad,self.wox_grad,self.bo_grad = (self.init_weight_gradient_mat())
        self.wch_grad,self.wcx_grad,self.bc_grad = (self.init_weight_gradient_mat())

        for t in range(self.times,0,-1):
            (wfh_grad,bf_grad,
            wih_grad,bi_grad,
            woh_grad,bo_grad,
            wch_grad,bc_grad) = (self.calc_gradient_t(t))

            self.wfh_grad += wfh_grad
            self.bf_grad += bf_grad
            self.wih_grad += wih_grad
            self.bi_grad += bi_grad
            self.woh_grad += woh_grad
            self.bo_grad += bo_grad
            self.wch_grad += wch_grad
            self.bc_grad += bc_grad
            print('---%d---' %t)
            print(wfh_grad)
            print(self.wfh_grad)

        xt = x.transpose()
        self.wfx_grad = np.dot(self.delta_f_list[-1],xt)
        self.wix_grad = np.dot(self.delta_i_list[-1],xt)
        self.wox_grad = np.dot(self.delta_o_list[-1],xt)
        self.wcx_grad = np.dot(self.delta_ct_list[-1],xt)
    
    def init_weight_gradient_mat(self):
        '''
        初始化权重矩阵
        '''
        wh_grad = np.zeros((self.state_width,self.state_width))
        wx_grad = np.zeros((self.state_width,self.input_width))
        b_grad = np.zeros((self.state_width,1))

        return wh_grad,wx_grad,b_grad

    def calc_gradient_t(self,t):
        '''
        计算每个时刻t权重的梯度
        '''
        h_prev = self.h_list[t-1].transpose()
        wfh_grad = np.dot(self.delta_f_list[t],h_prev)
        bf_grad = self.delta_f_list[t]#为什么
        wih_grad = np.dot(self.delta_i_list[t],h_prev)
        bi_grad = self.delta_f_list[t]
        woh_grad = np.dot(self.delta_o_list[t], h_prev)
        bo_grad = self.delta_f_list[t]
        wch_grad = np.dot(self.delta_ct_list[t], h_prev)
        bc_grad = self.delta_ct_list[t]

        return wfh_grad,bf_grad,wih_grad,bi_grad,woh_grad,bo_grad,wch_grad,bc_grad


    def update(self):

        self.wfh -= self.learning_rate * self.wfh_grad
        self.wfx -= self.learning_rate * self.wfx_grad
        self.bf -= self.learning_rate * self.bf_grad
        self.wih -= self.learning_rate * self.wih_grad
        self.wix -= self.learning_rate * self.wix_grad
        self.bi -= self.learning_rate * self.bi_grad
        self.woh -= self.learning_rate * self.woh_grad
        self.wox -= self.learning_rate * self.wox_grad
        self.bo -= self.learning_rate * self.bo_grad
        self.wch -= self.learning_rate * self.wch_grad
        self.wcx -= self.learning_rate * self.wcx_grad
        self.bc -= self.learning_rate * self.bc_grad 

    def reset_state(self):
        # 当前时刻初始化为t0
        self.times = 0       
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec()
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec()
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()

def data_set():

    x = [np.array([[1],[2],[3]]),np.array([[2],[3],[4]])]
    d = np.array([[1],[2]])

    return x,d      

def gradient_check():

    error_function  = lambda o:o.sum()

    lstm = LstmLayer(3,2,1e-3)

    x,d = data_set()
    lstm.forward(x[0])
    lstm.forward(x[1])

    sensitivity_array = np.ones(lstm.h_list[-1].shape,dtype = np.float64)

    lstm.backward(x[1],sensitivity_array,TanhActivator)

    epsilon = 10e-4
    for i in range(lstm.wfh.shape[0]):
        for j in range(lstm.wfh.shape[1]):
            lstm.wfh[i,j] += epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err1 = error_function(lstm.h_list[-1])
            lstm.wfh[i,j] -= 2*epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err2 = error_function(lstm.h_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            lstm.wfh[i,j] += epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e'%(i, j, expect_grad, lstm.wfh_grad[i,j]))
    return lstm

gradient_check()