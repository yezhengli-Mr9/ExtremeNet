import torch
print(torch.__version__)
from torch.autograd import Variable
w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)
w2 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)
print(w1.grad) 
print(w2.grad)
#------
d = torch.mean(w1)
d.backward()
w1.grad

w1.grad.data.zero_()
w1.grad