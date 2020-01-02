# Libraries
import os
import numpy as np
import sklearn as skl
import ast
import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from pandas import CategoricalDtype
from sklearn.metrics import recall_score


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
    medium = tracks['set', 'subset'] <= 'small'

    # create training and test set
    y_train = tracks.loc[medium & train, ('track', 'genre_top')]
    y_test = tracks.loc[medium & test, ('track', 'genre_top')]
    X_train = features.loc[medium & train, ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid']]
    X_test = features.loc[medium & test, ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid']]
    # to select only one feature (eg mfcc)
    # X_train = features.loc[small & train, 'mfcc']
    # X_test = features.loc[small & test, 'mfcc']

    # print stats
    print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
    print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

    # Be sure training samples are shuffled.
    X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance.
    scaler =skl.preprocessing.StandardScaler(copy=False)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    X = features.loc[medium]
    X = skl.decomposition.PCA(n_components=2).fit_transform(X)

    y = tracks.loc[medium, ('track', 'genre_top')]
    y = skl.preprocessing.LabelEncoder().fit_transform(y)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', alpha=0.5)
    # X.shape, y.shape
    plt.show()


    # Support vector classification with rbf kernel.
    clf = skl.svm.SVC(C=1.5, kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print("y_pred " + y_pred)
    score = clf.score(X_test, y_test)
    recall = recall_score(y_test, y_pred, average='macro')
    print('SVM rbf Accuracy: {:.2%}'.format(score))
    print('SVM rbf Recall: {:.2%}'.format(recall))

    # Support vector classification with linear kernel.
    clf = skl.svm.SVC(C=1.5, kernel='linear')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('SVM linear Accuracy: {:.2%}'.format(score))

    # KNN classification.
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=200)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('KNN Accuracy: {:.2%}'.format(score))

    #LR classification.
    clf = sklearn.linear_model.LogisticRegression(C=0.09)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('LR Accuracy: {:.2%}'.format(score))

    #Multi Class SVM classification.
    clf = sklearn.multiclass.OneVsRestClassifier(skl.svm.SVC()).fit(X_test, y_test)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Multiclass SVm Accuracy: {:.2%}'.format(score))
