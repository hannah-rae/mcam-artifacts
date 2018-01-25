import os
from glob import glob
from bokeh.plotting import figure, show, output_file
from scipy.optimize import curve_fit
from scipy.special import expit
import numpy as np


def plot_labels_and_loss():
    accept_dir = '/Users/hannahrae/src/mcam-artifacts/v2/labeler/data/supervised_labels/accept'
    retransmit_dir = '/Users/hannahrae/src/mcam-artifacts/v2/labeler/data/supervised_labels/retransmit'

    y = []
    X = []

    for example in glob(accept_dir + '/*'):
        ex_name = example.split('/')[-1][:-4]
        i, l, c = ex_name.split('_')
        y.append(float(1))
        X.append(float(l))

    for example in glob(retransmit_dir + '/*'):
        ex_name = example.split('/')[-1][:-4]
        i, l, c = ex_name.split('_')
        y.append(float(0))
        X.append(float(l))


    def lr(x, b, w):
        return expit(w*x + b)

    p_opt, p_cov = curve_fit(lr, X, y)
    print p_opt

    xs = np.arange(min(X), max(X), 0.0001)
    print xs
    ys = lr(xs, *p_opt)
    print ys


    p = figure(title='Logistic Regression for Label Prediction')
    p.xaxis.axis_label = 'Information Loss from Compression'
    p.yaxis.axis_label = 'P(label=accept)'
    #p.multi_line([xs, ys], [[1, 2, 3, 4], [1,2,3,4]])
    p.line(x=xs, y=ys)
    p.scatter(X, y)
    output_file("plotty.html")
    show(p)


if __name__ == '__main__':
    plot_labels_and_loss()