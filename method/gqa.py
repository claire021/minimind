import torch
import torch.nn as nn
dropout_layer = nn.Dropout(p=0.5) # 50%概率丢弃

t1 = torch.Tensor([1,2,3])
t2 = dropout_layer(t1)
# 丢弃是为了保持期望不变，将其他部分扩大两倍
print(t2)

layer = nn.Linear(in_features = 3, out_features=5, bias = True)
t1 = torch.Tensor([1,2,3]) # shape (3,)
t2 = torch.Tensor([[1,2,3]]) # shape(1,3)
# 这里的w和b是随机的，真实训练会在optimizer上更新
output2 = layer(t2) # shape(1,5)
print(output2)
# 线性变换：对应用的张量乘一个w矩阵然后加b

t = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]]) # [2,6]
t_view1 = t.view(3,4)
print(t_view1)
t_view2 = t.view(4,3)
# 改变视觉行列形状

t1 = torch.Tensor([[1,2,3], [4,5,6]]) # (2,3)
t1 = t1.transpose(0,1) # 交换行列

x = torch.tensor([1,2,3], [4,5,6],[7,8,9])
print(torch.triu(x, diagonal=0)) # 取上upper triangular 其余位置全部置为 0 diagonal 决定 从哪一条对角线开始保留
#diagonal = 0：主对角线及以上
#diagonal = 1：主对角线上方一条对角线开始

x = torch.arange(1,7)
y = torch.reshape(x,(2,3))
