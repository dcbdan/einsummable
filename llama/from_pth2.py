import torch
import struct
import numpy as np
import sys

BASE = "/data/30B/"
BIN  = "/data/es/30B"

if len(sys.argv) != 2:
  raise RuntimeError("incorrect num args")

i = int(sys.argv[1])

si = str(i)
if i < 10:
  si = "0" + si

inn_filename = BASE + "consolidated." + si + ".pth"
print("Reading from ", inn_filename)

weights = torch.load(inn_filename, "cpu", weights_only=True)

out_filename = BIN + "_" + si
print("Writing to ", out_filename)

with open(out_filename, "wb") as f:
  for key, value in weights.items():
    #print(key)
    text = key.ljust(50)
    binary_data = text.encode("utf-8")
    #print(key, value.shape, torch.sum(value), value.reshape(-1)[3])
    array = value.numpy()
    f.write(binary_data)
    f.write(np.array([value.numel()], dtype=np.int64).tobytes())
    f.write(array.tobytes())
