import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse

@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000
    noisy = noisy

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
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft),
                            onesided=True)
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,
                    help="path to model checkpoint")
parser.add_argument("--input", type=str,
                    help="path to noisy audio")
parser.add_argument("--output_dir", type=str,
                    help="where to save enhancd audio")

args = parser.parse_args()

if __name__ == '__main__':
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1)
    model_state_dict = torch.load(args.model_path, map_location=torch.device("cpu"))["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()
    enhance_one_track(model,
                      args.input,
                      args.output_dir,
                      16000*16,
                      n_fft,
                      n_fft//4, True)
