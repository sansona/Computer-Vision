import numpy as np
from sklearn import svm
from sklearn.externals import joblib

from image_enhancement import *

#------------------------------------------------------------------------------
# trained svc using various sudoku puzzles scraped from web. If OCR not
# working well, recommend retraining svc w/ new training data
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

    # makes feature & prediction vectors
    folders = os.listdir()
    for folder in folders:
        f_path = os.chdir(folder)

        files = [f for f in os.listdir(f_path)]
        # ensure feature vectors in (100x100), append to x_train
        for f in files:
            feature_data = LoadImage(f, grayscale=True)
            dx, dy = feature_data.shape
            if (dx, dy) != (100, 100):
                feature_data = resize(feature_data, (100, 100))

            x_train.append(feature_data)
            y_train.append(int(folder))

        os.chdir(PATH)

    # formats feature vectors to 100x100
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
    #print('Finished training')

#------------------------------------------------------------------------------


def SVCPredict(im, model_file='svc.joblib'):

    model = joblib.load(model_file)

    # formats im to proper shape - 100x100
    x = LoadImage(im, grayscale=True)
    nx, ny = x.shape
    if (nx, ny) != (100, 100):
        x = resize(x, (100, 100))
        nx, ny = x.shape
    x_pred = x.reshape((1, nx*ny))

    # so far a single SVC works really well for digit detection purposes. If
    # find errors in later testing, will update to ensemble method
    return model.predict(x_pred)

#------------------------------------------------------------------------------


# to train network
'''
x, y = FormatTrainingData('testing')
TrainSVC(x, y)
'''
