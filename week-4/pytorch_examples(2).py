#!/usr/bin/env python3
# file: pytorch_examples.py
import sys

import numpy as np
import torch

print("Num GPUs Available: ", torch.cuda.device_count())

# Try this code in console:
X = torch.arange(12, dtype=torch.float32)
x1=X.numel()
x2=X.shape
XR=X.reshape(3,4)
XR=torch.reshape(X,(3,4))
XR=torch.reshape(X,(3,-1))
x6=torch.zeros((3,4)) #zeros_like
x7=torch.ones((3,4)) #ones_like
x8=torch.randn((3,4))


y1=torch.exp(XR)
y2=XR.sum()
y3=XR.mean()
y4 = XR.sum(dim=0)
y5 = XR.mean(dim=0)  # also axis=0

x=torch.tensor([2,4,6,8])
y=torch.tensor([2,2,2,2])
z1=x/y
z2=x-y
z3=x**y
z4 = x*y
z4 = x.mul(y)
z4 = torch.mul(x,y)
# broadcasting
z5 = x*10
#sum through dim 0 (sums through rows)
z6 = XR/XR.sum(dim=0, dtype=torch.float32)


#tensors are generally immutable, but some opeartions can be inplace
q1=x
before=id(q1)
q1 = q1 + y
q2=before==id(q1)
before2=id(q1)
q1 += y
q3=before2==id(q1)


q4=x.to(dtype=torch.float32)
q4=x.float()

n1=np.arange(12)
q5=torch.tensor(n1)
q6=torch.from_numpy(n1)
n1[5]=100

n2=np.array(q5)
n2=np.asarray(q5)
n3=int(q5.sum())
n4=q5.sum().item()

r=torch.tensor(0.5, requires_grad=True)
s=torch.sin(r)
s.backward() # will compute gradient in leaves of computation graph
t1=r.grad
t1=r.grad.clone() #clone is generally differentiable, so gradients will flow back from the result of this operation to input
t1=torch.cos(r.detach()) #derivative of sin(r) with respect to r, the result of detach never requires gradient

# When working with tensors prefer using torch.Tensor.clone(),
# torch.Tensor.detach(), and torch.Tensor.requires_grad_() for readability.
# Letting t be a tensor, torch.tensor(t) is equivalent to t.detach().clone(),
# and torch.tensor(t, requires_grad=True) is equivalent to t.detach().clone().requires_grad_(True).

# torch.as_tensor() preserves autograd history and avoids copies where possible.
# torch.from_numpy() creates a tensor that shares storage with a NumPy array.

x=x.to(torch.float32)
x.requires_grad_(True)
z=torch.dot(x,x)
z.backward()
g1=x.grad
g2 = g1 + 0
g3 = g1.clone()

x.grad.zero_()  # optimizer.zero_grad()
z=torch.dot(x,x)
#z=torch.dot(x,x)+z.clone()
z=torch.dot(x,x)+z
z.backward()
g4=x.grad.clone()

x.grad.zero_()
z=torch.dot(x,x)
z=torch.dot(x,x)+z.detach()
z.backward()
g5=x.grad


#https://stackoverflow.com/questions/59560043/what-is-the-difference-between-model-todevice-and-model-model-todevice
# For tensor a: a.to("cuda") does not move the tensor! Must be a=a.to("cuda")
# For model a: a.to("cuda") moves the model to device inplace
# To copy a whole model:
# # a_copy = copy.deepcopy(a)
# # a_copy.to("cuda")

sys.exit(0)
