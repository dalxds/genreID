import os
from os.path import normpath, basename
import librosa
import librosa.display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils


def full_frame(width=None, height=None):
    # remove padding from figures
    mpl.rcParams['savefig.pad_inches'] = 0
    # resize figure
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    # remove axis
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # set autoscale to tight
    plt.autoscale(tight=True)


def get_output_path(audio_path):
    # clear path string
    trim_path = basename(normpath(audio_path))
    # take track id from path
    track_id = trim_path[0:6]
    # find dataset split for track
    split = splits[int(track_id)]
    return OUTPUT_PATH + '/' + split + '/' + track_id + '.png'


def generate_mel_spec(input_path, output_path):
    # TODO Handle exceptions
    # input audio
    y, sr = librosa.load(input_path, mono=True, duration=30.0)
    # generate mel spectogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=2048 // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # apply custom frame
    full_frame()
    # save figure
    librosa.display.specshow(S_dB, sr=sr)
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    if __debug__:
        debug_count = 0

    # read tracks' data
    tracks = utils.load('')  # [ADD] Path to metadata file
    # select FMA Small
    subset = tracks.index[tracks['set', 'subset'] <= 'small']
    # Create dataframe with the dataset splits (training, validation, testing)
    splits = tracks.loc[subset, ('set', 'split')]

    # import folder
    PATH_TO_FMA = ""  # [ADD] path to FMA Small audio data

    # create output folders
    OUTPUT_PATH = PATH_TO_FMA + "/../melspecs_export"
    training_path = OUTPUT_PATH + '/' + "training"
    validation_path = OUTPUT_PATH + '/' + "validation"
    test_path = OUTPUT_PATH + '/' + "test"

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        os.makedirs(training_path)
        os.makedirs(validation_path)
        os.makedirs(test_path)

    os.chdir(PATH_TO_FMA)

    for folder in os.listdir(PATH_TO_FMA):
        folder_path = PATH_TO_FMA + "/" + folder
        if not os.path.isdir(folder_path):
            continue
        os.chdir(folder_path)
        for file in os.listdir(folder_path):
            audio_path = os.path.abspath(file)
            mel_path = get_output_path(audio_path)
            mel_spec_exists = os.path.exists(mel_path)
            if not mel_spec_exists:
                generate_mel_spec(audio_path, mel_path)
            if __debug__:
                debug_count += 1
                print("Count: {0}/8000 --- E Status: {1} --- Filename: {2}".format(debug_count, mel_spec_exists, mel_path))
