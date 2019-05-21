import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision 包收录了若干重要的公开数据集、网络模型和计算机视觉中的常用图像变换
import torchvision
import torchvision.transforms as transforms  

import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)    # Sets the seed for generating random numbers.reproducible
# 10个数据点 数据点少的情况才可以凸显拟合问题
N_SAMPLES = 20
N_HIDDEN = 300

# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
print('x.size()',x.size())

# torch.normal(mean, std, out=None) → Tensor
y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# show data
# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# plt.legend(loc='upper left')
# plt.ylim((-2.5, 2.5))
# plt.show()

#我们现在搭建两个神经网络, 一个没有 dropout, 一个有 dropout. 没有 dropout 的容易出现 过拟合, 那我们就命名为 net_overfitting, 另一个就是 net_dropped.
net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1,N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,1),
)

net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1,N_HIDDEN),
    torch.nn.Dropout(0.5), # 0.5的概率失活
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,1),
)

# 训练模型并测试2个模型的performance
optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(),lr=0.001)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(),lr=0.001)
loss = torch.nn.MSELoss()

for epoch in range(500):
    pred_ofit= net_overfitting(x)
    pred_drop= net_dropped(x)
    
    loss_ofit = loss(pred_ofit,y)
    loss_drop = loss(pred_drop,y)
    
    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    
    loss_ofit.backward()
    loss_drop.backward()
    
    optimizer_ofit.step()
    optimizer_drop.step()
    
    if epoch%100 ==0 :
        net_overfitting.eval() # 将神经网络转换成测试形式,此时不会对神经网络dropout
        net_dropped.eval() # 此时不会对神经网络dropout
        
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        
        # show data
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss(test_pred_ofit, test_y).data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss(test_pred_drop, test_y).data.numpy(), fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)
        plt.ioff()
        plt.show()
        
        net_overfitting.train()
        net_dropped.train()
        



