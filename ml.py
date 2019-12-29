# Libraries
import os
import numpy as np
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import IPython.display as ipd

# Files
import utils

if __name__ == '__main__':
    # Load metadata and features.
    tracks = utils.load('src/fma_metadata/tracks.csv')
    genres = utils.load('src/fma_metadata/genres.csv')
    features = utils.load('src/fma_metadata/features.csv')

    # assert and print shape
    np.testing.assert_array_equal(features.index, tracks.index)

    # print shapes
    print(tracks.shape, genres.shape, features.shape)

    # print heads
    # ipd.display(tracks['track'].head())
    # ipd.display(tracks['album'].head())
    # ipd.display(tracks['artist'].head())
    # ipd.display(tracks['set'].head())

    train = tracks['set', 'split'] == 'training'
    val = tracks['set', 'split'] == 'validation'
    test = tracks['set', 'split'] == 'test'

    # use only small dataset
    small = tracks['set', 'subset'] <= 'small'

    # create training and test set
    y_train = tracks.loc[small & train, ('track', 'genre_top')]
    y_test = tracks.loc[small & test, ('track', 'genre_top')]
    X_train = features.loc[small & train]
    X_test = features.loc[small & test]
    # to select only one feature (eg mfcc)
    # X_train = features.loc[small & train, 'mfcc']
    # X_test = features.loc[small & test, 'mfcc']

    # print stats
    print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
    print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

    # Be sure training samples are shuffled.
    # X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

    # Preprocessing
    # Standardize features by removing the mean and scaling to unit variance.
    # scaler = skl.preprocessing.StandardScaler(copy=False)
    # scaler.fit_transform(X_train)
    # scaler.transform(X_test)
    #
    # # Support vector classification
    # clf = skl.svm.SVC()
    # clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    # print('Accuracy: {:.2%}'.format(score))
