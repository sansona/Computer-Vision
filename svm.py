import numpy as np
from sklearn import svm

from image_enhancement import *


def FormatTrainingData(training_im_folder):
    '''
    takes in directory of digits, formats into x_train & y_train numpy arrays
    '''

    im_extensions = ['.jpg', '.jpeg', '.png', '.tif']
    PATH = os.getcwd() + '/' + training_im_folder
    os.chdir(PATH)

    x_train = []
    y_train = []

    folders = os.listdir()
    for folder in folders:
        f_path = os.chdir(folder)
        files = [f for f in os.listdir(f_path)]
        for f in files:
            x_train.append(LoadImage(f))
            y_train.append(int(folder))
        os.chdir(PATH)

    # need to get x_train in 2D
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    assert len(x_train == len(y_train))
    return x_train, y_train


def TrainSVC(x_train, y_train):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
