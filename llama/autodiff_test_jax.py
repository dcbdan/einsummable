import numpy as np

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn

def softmax_test():
  @grad
  def f(xy):
    x,y = xy
    return jnp.sum(nn.softmax(x)*y)

  x = jnp.array([0.9,0.3,-0.3,0.1]).reshape((2,2))
  y = jnp.array([0.1,0.2,0.3,0.4]).reshape((2,2))
  gx,gy = f((x,y))
  print("gx: ", gx.reshape(-1))
  print("gy: ", gy.reshape(-1))

def softmax_test3():
  print("softmax_test3")
  @grad
  def f(xy):
    x,y = xy
    return jnp.sum(jnp.exp(x)*y)

  x = jnp.array([0.9,0.3,-0.3,0.1]).reshape((2,2))
  y = jnp.array([0.1,0.2,0.3,0.4]).reshape((2,2))
  gx,gy = f((x,y))
  print("gx: ", gx.reshape(-1))
  print("gy: ", gy.reshape(-1))

softmax_test()
