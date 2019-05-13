# import torch
# x = torch.rand(2, 2, requires_grad=True)
# learning_rate =0.1 #学习率
# epoches =5 #学习周期

# for epoch in range(epoches):
#      y = x**2+2*x+1
#      y.backward(torch.ones_like(x))
#      print("grad",x.grad.data) #x的梯度值
#      x.data = x.data - learning_rate*x.grad.data #更新x
#      x.grad.data.zero_()
# print(x.data)

# import torch 
# #print(torch.__version__)

# # train data 
# x_data= torch.arange(1.0,4.0,1.0)
# print(x_data)
# x_data=x_data.view(-1,1)
# print(x_data)
# y_data= torch.arange(2.0,7.0,2.0)
# print(y_data)
# y_data= y_data.view(-1,1)
# print(y_data)

# # 超参数设置
# learning_rate=0.1
# num_epoches=40

# # 线性回归模型
# class LinearRegression(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = torch.nn.Linear(1,1)# 1 in and 1 out
        
#     def forward(self,x):
#         y_pred = self.linear(x)
#         return y_pred

# model = LinearRegression()

# # 定义loss function损失函数和optimizer优化器
# # PyTorch0.4以后，使用reduction参数控制损失函数的输出行为
# criterion = torch.nn.MSELoss(reduction='mean')
# # nn.Parameter - 张量的一种，当它作为一个属性分配给一个Module时，它会被自动注册为一个参数。
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# # 训练模型
# for epoch in range(num_epoches):
#     # forward 
#     y_pred= model(x_data)
    
#     #computing loss 
#     loss = criterion(y_pred,y_data)
    
#     print(epoch,'epoch\'s loss:',loss.item())
    
#     # backward: zero gradients + backward + step
#     optimizer.zero_grad()
#     loss.backward()  
#     optimizer.step() # 执行一步-梯度下降（1-step gradient descent）
    
# # testing
# x_test=torch.Tensor([4.0])
# print("the result of y when x is 4:",model(x_test))
# print('model.parameter:',list(model.parameters()))

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt

NUM = 100   #输入个数（输入层神经元个数）
hider_num = 300  #隐藏层神经元个数
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
 
        self.fc1 = nn.Linear(NUM,hider_num)
        self.fc2 = nn.Linear(hider_num,NUM)
 
    def forward(self,x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x = torch.randn(NUM)
input = Variable(x)   #随机生成NUM个数据
 
target = Variable(0.5 * x + 0.3)   #用0.5 × x + 0.3 函生成目标数据

net = Net()   #网络对象
print(net)
 
optimizer = optim.SGD(net.parameters(),lr=0.01)  #随机梯度下降优化器
loss_list =[]                                #保存loss，便于画图
step = 500                                    #迭代次数
 
for epoch in range(step):
    optimizer.zero_grad()                    #参数梯度清零，因为会累加
    out = net(input)                         #通过一次网络的输出
    loss = nn.MSELoss()(out,target)           #计算输出与target数据的均方差
    loss_list.append(loss)                    #保存loss
    loss.backward()                           #loss反向传播
    optimizer.step()                          #更新参数w，b

plt.figure(1)
plt.plot(range(1,NUM+1),target.detach().numpy().tolist(),'*',ms=10,lw=1,color='black')
plt.plot(range(1,NUM+1),out.detach().numpy().tolist(),'o',ms=3,lw=1,color='red')
plt.show()   #画出target和输出的位置图
plt.figure(2)
plt.plot(range(1,step+1),loss_list,'o-',ms=3,lw=1,color='black')
plt.show()   #画loss图