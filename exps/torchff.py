import torch
from torch.autograd import Variable

niter = 10
dn = 10000
dp = 100
dd = 10
dws = [2000,200,200]
learning_rate = 0.01

def rand(d0,d1):
  return 0.1 * (torch.rand(d0,d1)-0.5)

X = 0.1*(rand(dn,dp)-0.5)
Y = 0.1*(rand(dn,dd)-0.5)
WS = []
_dws = [dp] + dws + [dd]
for da,db in zip(_dws[:-1], _dws[1:]):
  WS.append(Variable(rand(da,db), requires_grad=True))


###

import time
start_time = time.time()

for i in range(niter):
  Yhat = torch.matmul(X, WS[0])
  if len(WS) == 1:
    pass
  else:
    Yhat = torch.relu(Yhat)
    for W in WS[1:-1]:
      Yhat = torch.matmul(Yhat, W)
      Yhat = torch.relu(Yhat)
    Yhat = torch.matmul(Yhat, WS[-1])
  loss = torch.sum(torch.square(Yhat - Y))
  if i % 75 == 0:
    print(loss.item())
  loss.backward()
  for W in WS:
    W.data -= learning_rate * W.grad.data
  for W in WS:
    W.grad.data.zero_()

end_time = time.time()
print("Time: ", end_time - start_time)



