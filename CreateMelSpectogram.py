import librosa
import librosa.display
import numpy as np
import os
from os.path import normpath, basename
import matplotlib as mpl
import matplotlib.pyplot as plt


def full_frame(width=13.66, height=0.96):
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)


def generate_mel_spec(fpath):
    y, sr = librosa.load(fpath, mono=True, duration=30.0)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=2048 // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)
    full_frame()
    librosa.display.specshow(S_dB, sr=sr)
    trim_path = basename(normpath(filepath))
    fpath = trim_path[0:6]
    plt.savefig(output_path + '/' + fpath + '.png')


if __name__ == '__main__':
    PATH_TO_FMA = "/Users/Dimitris/Downloads/AVT/fma_small_1"

    output_path = PATH_TO_FMA + "/../melspecs_output_1"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path = PATH_TO_FMA
    os.chdir(path)

    for folder in os.listdir(path):
        folder_path = path + "/" + folder
        if not os.path.isdir(folder_path):
            continue
        os.chdir(folder_path)
        for file in os.listdir(folder_path):
            print(file)
            file_path = os.path.abspath(j)
            generate_mel_spec(file_path)