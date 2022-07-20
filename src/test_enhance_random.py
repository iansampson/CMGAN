import time
import numpy as np
import torch
from models import generator
from utils import *

cut_len = 16000 * 16
n_fft = 400
hop = 100

noisy = torch.rand(1, 32000) # 2 seconds
model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1)

c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
noisy = torch.transpose(noisy, 0, 1)
noisy = torch.transpose(noisy * c, 0, 1)

length = noisy.size(-1)
frame_num = int(np.ceil(length / 100))
padded_len = frame_num * 100
padding_len = padded_len - length
noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
if padded_len > cut_len:
    batch_size = int(np.ceil(padded_len/cut_len))
    while 100 % batch_size != 0:
        batch_size += 1
    noisy = torch.reshape(noisy, (batch_size, -1))

noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft), onesided=True)
noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)

t0 = time.process_time()
est_real, est_imag = model(noisy_spec)
t1 = time.process_time()
print("Evaluation time in seconds:", t1 - t0)

# print("est_real", est_real.shape)
# print("est_imag", est_imag.shape)
