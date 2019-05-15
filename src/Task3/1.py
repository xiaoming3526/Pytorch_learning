import torch
import torch.nn as nn
import numpy as np

#我们这里使用的 german.data-numeric是numpy处理好数值化数据，我们直接使用numpy的load方法读取即可
data=np.loadtxt("../german.data-numeric")
#数据读取完成后我们要对数据做一下归一化的处理
n,l=data.shape
for j in range(l-1):
    meanVal=np.mean(data[:,j])
    stdVal=np.std(data[:,j])
    data[:,j]=(data[:,j]-meanVal)/stdVal

#打乱数据
np.random.shuffle(data)


# 区分训练集和测试集，由于这里没有验证集，所以我们直接使用测试集的准确度作为评判好坏的标准
# 区分规则：900条用于训练，100条作为测试
# german.data-numeric的格式为，前24列为24个维度，最后一个为要打的标签（0，1），所以我们将数据和标签一起区分出来
train_data=data[:900,:l-1]
train_lab=data[:900,l-1]-1
test_data=data[900:,:l-1]
test_lab=data[900:,l-1]-1


class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.fc=nn.Linear(24,2) # 由于24个维度已经固定了，所以这里写24
    def forward(self,x):
        out=self.fc(x)
        out=torch.sigmoid(out)
        return out

#测试集上的准确率
def test(pred,lab):
    t=pred.max(-1)[1]==lab
    return torch.mean(t.float())

net=LR() 
criterion=nn.CrossEntropyLoss() # 使用CrossEntropyLoss损失
optm=torch.optim.Adam(net.parameters()) # Adam优化
epochs=1000 # 训练1000次

for i in range(epochs):
    # 指定模型为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    x=torch.from_numpy(train_data).float()
    y=torch.from_numpy(train_lab).long()
    y_hat=net(x)
    loss=criterion(y_hat,y) # 计算损失
    optm.zero_grad() # 前一步的损失清零
    loss.backward() # 反向传播
    optm.step() # 优化
    if (i+1)%100 ==0 : # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in=torch.from_numpy(test_data).float()
        test_l=torch.from_numpy(test_lab).long()
        test_out=net(test_in)
        # 使用我们的测试函数计算准确率
        accu=test(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i+1,loss.item(),accu))