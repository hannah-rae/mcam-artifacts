import os

from bokeh.plotting import figure, show, output_file
from scipy.optimize import curve_fit
from scipy.special import expit
import numpy as np


def plot_labels_and_loss():
    filenames = ['labeler/data/%s/metadata.txt' % d for d in os.listdir('labeler/data/') if d.startswith('labeled')]

    labels = []
    losses = []

    def label_s_to_i(label):
        if label == 'a':
            return 1
        else:
            return 0

    for filename in filenames:
        for line in file(filename):
            i, l, c, lbl = line.split()
            labels.append(float(label_s_to_i(lbl)))
            losses.append(float(l))


    def lr(x, b, w):
        return expit(w*x + b)

    p_opt, p_cov = curve_fit(lr, losses, labels)
    print p_opt

    xs = np.arange(min(losses), max(losses), 0.0001)
    ys = lr(xs, *p_opt)


    p = figure(title='Logistic Regression for Label Prediction')
    p.xaxis.axis_label = 'Information Loss from Compression'
    p.yaxis.axis_label = 'P(label=accept)'
    p.line(x=xs, y=ys)
    p.scatter(losses, labels)
    output_file("plotty.html")
    show(p)


if __name__ == '__main__':
    plot_labels_and_loss()