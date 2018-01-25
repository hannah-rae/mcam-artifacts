from glob import glob
from scipy.special import expit
import numpy as np

# Test directories
accept_dir = '/Users/hannahrae/src/mcam-artifacts/v2/labeler/data/supervised_labels/accept_test'
retransmit_dir = '/Users/hannahrae/src/mcam-artifacts/v2/labeler/data/supervised_labels/retransmit_test'

def predict_1d(x, theta=-1.38705672, b=14.83335514):
    return 1 / (1 + np.exp(-1 * (np.dot(theta, x)+b)))

def predict_2d(x, theta=np.array([-1.20730291, 0.15657625]), b=0):
    return 1 / (1 + np.exp(-1 * (np.dot(theta, x)+b)))

correct_1d = 0
correct_2d = 0

for example in glob(accept_dir + '/*'):
    ex_name = example.split('/')[-1][:-4]
    i, l, c = ex_name.split('_')
    y_1d = predict_1d(x=float(l))
    y_2d = predict_2d(x=np.array([float(l), float(c)]))
    print y_1d, y_2d

    if y_1d >= 0.5:
        correct_1d += 1

    if y_2d >= 0.5:
        correct_2d += 1

for example in glob(retransmit_dir + '/*'):
    ex_name = example.split('/')[-1][:-4]
    i, l, c = ex_name.split('_')
    y_1d = predict_1d(x=float(l))
    y_2d = predict_2d(x=np.array([float(l), float(c)]))
    print y_1d, y_2d

    if y_1d < 0.5:
        correct_1d += 1

    if y_2d < 0.5:
        correct_2d += 1

acc_1d = float(correct_1d) / 42.
acc_2d = float(correct_2d) / 42.
print "Accuracy"
print "1D model: %f" % acc_1d
print "2D model: %f" % acc_2d