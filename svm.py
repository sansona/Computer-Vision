import numpy as np
from sklearn import svm
from sklearn.externals import joblib

from image_enhancement import *


#------------------------------------------------------------------------------
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
            x_train.append(LoadImage(f, grayscale=True))
            y_train.append(int(folder))
        os.chdir(PATH)

    # need to get x_train in 2D. Should be 100x100
    x = np.array(x_train)
    n_im, nx, ny = x.shape

    x_train = x.reshape((n_im, nx*ny))
    y_train = np.array(y_train)

    assert len(x_train) == len(y_train)
    return x_train, y_train

#------------------------------------------------------------------------------


def TrainSVC(x_train, y_train):
    os.chdir('..')
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    model = joblib.dump(clf, 'svc.joblib')
    print('Trained classifier')

#------------------------------------------------------------------------------


def SVCPredict(im, model_file='svc.joblib'):

    model = joblib.load(model_file)
    x = LoadImage(im, grayscale=True)
    nx, ny = x.shape
    x_pred = x.reshape((1, nx*ny))
    return model.predict(x_pred)

#------------------------------------------------------------------------------


# example usage
'''
x, y = FormatTrainingData('testing')
TrainSVC(x, y)
print(SVCPredict('square10.png'))
'''
