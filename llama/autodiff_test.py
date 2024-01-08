from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024

class RMSNorm(torch.nn.Module):
  def __init__(self, w, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(w)

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )

        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )

        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cpu()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cpu()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, zz = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

#class AttentionDouble(nn.Module):
#    def __init__(self, args: ModelArgs):
#        super().__init__()
#
#        self.n_local_heads = args.n_heads
#        self.head_dim = args.dim // args.n_heads
#
#        self.wq = nn.Linear(
#            args.dim,
#            args.n_heads * self.head_dim,
#            bias=False,
#            dtype=torch.float64
#        )
#
#        self.wk = nn.Linear(
#            args.dim,
#            args.n_heads * self.head_dim,
#            bias=False,
#            dtype=torch.float64
#        )
#
#        self.wv = nn.Linear(
#            args.dim,
#            args.n_heads * self.head_dim,
#            bias=False,
#            dtype=torch.float64
#        )
#
#        self.wo = nn.Linear(
#            args.n_heads * self.head_dim,
#            args.dim,
#            bias=False,
#            dtype=torch.float64
#        )
#
#        self.cache_k = torch.zeros(
#            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
#        ).cpu()
#        self.cache_v = torch.zeros(
#            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
#        ).cpu()
#
#    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
#        bsz, seqlen, zz = x.shape
#        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
#
#        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#
#        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
#
#        self.cache_k = self.cache_k.to(xq)
#        self.cache_v = self.cache_v.to(xq)
#
#        self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
#        self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv
#
#        keys = self.cache_k[:bsz, : start_pos + seqlen]
#        values = self.cache_v[:bsz, : start_pos + seqlen]
#
#        xq = xq.transpose(1, 2)
#        keys = keys.transpose(1, 2)
#        values = values.transpose(1, 2)
#        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
#        if mask is not None:
#            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
#        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
#        output = output.transpose(
#            1, 2
#        ).contiguous().view(bsz, seqlen, -1)
#
#        return self.wo(output)

#############################################################################
def print_as_vector(name, t):
  fix = lambda s: str(float(s))
  print("vector<float> " + name + "_{" + ",".join(map(fix, t.reshape(-1))) + "};")

def rms_norm_test(x,weight):
  norm = RMSNorm(weight)
  y = norm.forward(x)
  sy = y.sum()
  sy.backward()
  print(norm.weight.grad)

def attention_test():
  args = ModelArgs(dim = 8, n_heads = 2, max_batch_size = 1, max_seq_len = 3)
  attention = Attention(args)
  freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
  print(freqs_cis.shape)
  wq = list(attention.wq.parameters())[0]
  wk = list(attention.wk.parameters())[0]
  wv = list(attention.wv.parameters())[0]
  wo = list(attention.wo.parameters())[0]
  print_as_vector("wq", wq)
  print_as_vector("wk", wk)
  print_as_vector("wv", wv)
  print_as_vector("wo", wo)
  x = torch.rand(1, 3, 8)
  print_as_vector("x", x)
  print("AAAAAAAAAAAAAAA")
  y = attention.forward(x, 0, freqs_cis, None)
  print("BBBBBBBBBBBBBBB")
  sy = y.sum()
  sy.backward()
  print("grad_wq", wq.grad.sum())
  print("grad_wk", wk.grad.sum())
  print("grad_wv", wv.grad.sum())
  print("grad_wo", wo.grad.sum())

def softmax_test():
  x = nn.Parameter(-0.5 + torch.rand(2,2))
  y = nn.Parameter(       torch.rand(2,2))
  z = F.softmax(x, dim=-1) * y;
  sz = z.sum()
  sz.backward()
  print_as_vector("x", x)
  print_as_vector("y", y)
  print(x.grad)
  print(y.grad)
  print("gx", x.grad.sum())
  print("gy", y.grad.sum())

def reduction_test():
  x = nn.Parameter(10 + torch.rand(5,5))
  y = nn.Parameter(10 + torch.rand(5,5))
  z = torch.matmul(x,y)
  print("xy:\n", z)
  #z = z.sum(axis=-1)
  #print("sum(xy, axis=-1):\n", z)
  z = z.max(axis=-1)[0]
  print("max(xy, axis=-1):\n", z)
  sz = z.sum()
  sz.backward()
  print_as_vector("x", x)
  print_as_vector("y", y)
  #print(x.grad)
  #print(y.grad)
  print("gx sum ", x.grad.sum())
  print("gy sum ", y.grad.sum())

def reduction_exp1():
  x = nn.Parameter(torch.rand(3,3))
  print("x\n",x)
  z, _ = x.max(dim=-1)
  print("x reduced\n",z)
  sz = z.sum()
  sz.backward()
  print("grad x", x.grad)

def reduction_exp2():
  x = nn.Parameter(torch.ones(3,3))
  print("x\n",x)
  z, _ = x.max(dim=-1)
  print("x reduced\n",z)
  sz = z.sum()
  sz.backward()
  print("grad x", x.grad)

def max_exp():
  # Note: t1.grad or t2.grad will be None which is all zeros.
  # ... We can't do that in our graph since we don't have a way
  #     to pass in constant zeros and then do a different simplify
  #     op

  t1 = torch.rand(10, requires_grad=True)
  t2 = torch.rand(10, requires_grad=True)

  s1 = torch.sum(t1)
  s2 = torch.sum(t2)
  print('sum t1:', s1, 'sum t2:', s2)
  m = max(s1, s2)
  print('max:', m, 'requires_grad:', m.requires_grad)
  m.backward()
  print('t1 gradients:', t1.grad)
  print('t2 gradients:', t2.grad)

def softmax_exp():
  x = nn.Parameter(-0.5 + torch.rand(5,dtype=torch.float64))
  y = torch.rand(5)
  z = F.softmax(x, dim=-1) * y
  z.sum().backward()
  print(x)
  print(z)
  print(x.grad)

def complex_test():
  x = nn.Parameter(-0.5 + torch.rand(1,2,dtype=torch.float32))
  y = torch.rand(1,dtype = torch.complex64)
  s = torch.view_as_real(torch.view_as_complex(x) * y).sum()
  s.backward()
  #print_as_vector("x", x)
  #print_as_vector("y", torch.view_as_real(y))
  print("(x.to_complex() * y).view_as_real().sum()")
  print(y)
  print(x.grad)
  print("")

def complex_test2():
  d = 1
  x = nn.Parameter(-0.5 + torch.rand(d,dtype=torch.complex64))
  y = torch.rand(d,dtype = torch.complex64)
  s = torch.sum(torch.view_as_real(x*y))
  s.backward()
  print("(x*y).view_as_real().sum()")
  print("y:  ", y)
  print("gx: ", x.grad)
  print("")

def complex_test3():
  x = nn.Parameter(-0.5 + torch.rand(1,dtype=torch.complex64))
  y = torch.rand(1,dtype = torch.complex64)
  s = torch.sum(x*y)
  s.backward()
  print("(x:complex*y).sum()")
  print("y:  ", y)
  print("gx: ", x.grad)
  print("")

def complex_test4():
  x = nn.Parameter(-0.5 + torch.rand(2))
  y = torch.view_as_complex(x)
  y.sum().backward()
  print(x.grad) # [1,0])
def complex_test5():
  print("x.view_as_real().sum()")
  x = torch.rand(1, dtype = torch.complex64)
  print("x:   ", x)
  x = nn.Parameter(x)
  y = torch.view_as_real(x)
  y.sum().backward()
  print("xg:  ", x.grad) # (1 + 1j)
def complex_test6():
  print("")
  print("x.view_as_complex().view_as_real().sum()")
  x = torch.rand(2, dtype = torch.float32)
  print("x:   ", x)
  x = nn.Parameter(x)
  y = torch.view_as_complex(x)
  y = torch.view_as_real(y)
  y.sum().backward()
  print("xg:  ", x.grad)
def complex_test7():
  print("")
  print("(x.view_as_complex()*y).view_as_real().sum()")
  x = torch.rand(2, dtype = torch.float32)
  y = torch.view_as_complex(torch.tensor([5,3], dtype=torch.float32))
  print("x:   ", x)
  print("y:   ", y)
  x = nn.Parameter(x)

  xc = torch.view_as_complex(x)
  xc.retain_grad()

  xcy = xc*y
  xcy.retain_grad()

  z = torch.view_as_real(xcy)
  z.retain_grad()

  zs = z.sum()
  zs.retain_grad()
  zs.backward()

  print("g zs :",zs.grad)
  print("g z  :",z.grad)
  print("g xcy:",xcy.grad)
  print("g xc :",xc.grad)
  print("g x  :",x.grad)

def softmax_test2():
  x = torch.tensor([0.9,0.3,-0.3,0.1]).reshape((2,2))
  y = torch.tensor([0.1,0.2,0.3,0.4]).reshape((2,2))

  x = nn.Parameter(x)
  y = nn.Parameter(y)

  z = F.softmax(x, dim=-1) * y;

  sz = z.sum()

  sz.backward()

  print("gx", x.grad)
  print("gy", y.grad)

def softmax_test3():
  print("softmax_test3")
  x = torch.tensor([0.9,0.3,-0.3,0.1]).reshape((2,2))
  y = torch.tensor([0.1,0.2,0.3,0.4]).reshape((2,2))

  x = nn.Parameter(x)
  y = nn.Parameter(y)

  z = torch.exp(x) * y;

  sz = z.sum()

  sz.backward()

  print("gx", x.grad)
  print("gy", y.grad)

def forward_test1(dn, dp, dd, dw1, dw2):
  x = torch.rand(dn,dp)
  y = torch.rand(dn,dd)
  w0 = nn.Parameter(torch.rand(dp,dw1))
  w1 = nn.Parameter(torch.rand(dw1,dw2))
  w2 = nn.Parameter(torch.rand(dw2,dd))

  print_as_vector("x", x);
  print_as_vector("y", y);
  print_as_vector("w0", w0)
  print_as_vector("w1", w1)
  print_as_vector("w2", w2)

  x = F.relu(torch.matmul(x, w0))
  x = F.relu(torch.matmul(x, w1))
  x = torch.matmul(x, w2)

  loss = torch.sum(torch.square(x-y))
  loss = loss / (dn*dd)

  loss.backward()
  print("gw0", w0.grad)
  print("gw1", w1.grad)
  print("gw2", w2.grad)

#reduction_test()
#reduction_exp2()
#max_exp()
#softmax_test2()
#softmax_exp()
#attention_test()
#complex_test()
#complex_test2()
#complex_test3()
#complex_test4()
#complex_test5()
#complex_test7()

forward_test1(4, 2, 2, 2, 2)
