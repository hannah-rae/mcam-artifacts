import random
import time
import threading
import os
import functools
import math
from glob import glob

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import dataset.mcam1
import dataset.mcam_image.local
import utils.image
import learner.mcam1

# This script takes one slice of a Mastcam image and obtains the output of the CNN
# for that input N times. The model then plots those predictions as a histogram.

N = 1000
# img_path = './bayes_test/3/7_85_10.572259_0.997018.png'
# img_path = './bayes_test/3/32_85_12.961396_0.027770.png'
img_path = './bayes_test/3/50_85_12.824570_0.353577.png'
# convert image to numpy array
i = Image.open(img_path)
img = np.array(i) / 256.

test_model = learner.mcam1.McamLearner(window_size=160, kp=0.1, savefile='saved_sessions/saved_1488556921_10535.meta')
preds = []
for n in range(N):
    # get the output of the model
    [[p, _]] = test_model([img])
    preds.append(p)

mu = np.mean(preds, axis=0)
print mu
var = np.var(preds, axis=0)
print var
pdf_x = np.linspace(np.min(preds),np.max(preds),N)
pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-mu)**2/var)

plt.figure()
plt.hist(preds, normed=True, color='grey')
plt.plot(pdf_x, pdf_y, 'k--')
plt.xlabel('Probability that a Scientist Finds the Image Quality Acceptable')
plt.ylabel('Normalized number of occurrences for N = 1000')
plt.title('Outputs with Predictive Mean and Uncertainty from Dropout Network')
plt.show()