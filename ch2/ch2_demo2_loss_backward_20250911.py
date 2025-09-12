'''
Implement the Neuro_Nodes in PyTorch format.
'''

import torch
import torch.nn as nn
import numpy    as np
import matplotlib.pyplot as plt

class Neuron(nn.Module):
    def __init__(self, input_size, output_size):
        super(Neuron, self).__init__()
        self.w = torch.randn(input_size, output_size, 
                 dtype=torch.float32, requires_grad=True)
        self.b = torch.randn(output_size, 
                 dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        #loss = (y_pred - y).pow(2).mean()
        return x @ self.w + self.b

    def update(self, lr=0.01):
        with torch.no_grad():
            self.w -= lr * self.w.grad
            self.b -= lr * self.b.grad

    def zero_grad(self): # reset the gradient to zero
        self.w.grad.zero_()
        self.b.grad.zero_()



if __name__=='__main__':
    # 单个神经元实现一元线性回归
    # 身高x与体重y的线性关系
    # y = w*x + b
    x = np.array([153.3, 164.9, 168.1, 151.5, 157.8, 156.7, 161.1])  # 身高
    y = np.array([45.5, 56.0, 55.0, 52.8, 55.6, 50.8, 56.4])         # 体重
    neuron_node = Neuron(1, 1) # 1个输入，1个输出的神经元节点
    assert (len(x) == len(y))  # 确保x和y的长度一致
    # x,y的散点图
    plt.scatter(x, y)
    lr = 0.00001  # learning rate
    X = torch.tensor(x, dtype=torch.float32) #convert np_array to tensor
    X = X.reshape(-1, 1) #reshape to 2D tensor, [Batch, dim]
    Y = torch.tensor(y, dtype=torch.float32) #convert np_array to tensor
    for ep in range(100):  # iterate 10 epoches

        # gradient descent的三部曲
        y_pred = neuron_node(X)            # 1. forward prediction
        loss = (y_pred - Y).pow(2).mean()  # 1. real mean square error loss

        loss.backward()                    # 2. backward gradient, compute the gradient of loss with respect to all the learnable parameters
        neuron_node.update(lr)             # 3. update weights and bias with learning rate
        neuron_node.zero_grad()            # 3. reset the gradient to zero
        print('epoch=', ep, 'w=', neuron_node.w.item(), 'b=', neuron_node.b.item(), 'mse=', loss.item())

    w = neuron_node.w.item() #convert tensor to scalar
    b = neuron_node.b.item() #convert tensor to scalar
    yp = y_pred.detach().numpy() #convert tensor to np_array
    loss = torch.sum(((y_pred - Y) ** 2)) / 2
    plt.plot(x, yp, label='epoch=' + str(ep) + ' mse loss=%.2f' % loss.item())
    print('epoch=', ep, 'w=', w, 'b=', b, 'mse=', loss.item())
    plt.legend()
    plt.show()
