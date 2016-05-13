
import random
import time
import threading
import os
import functools

from PIL import Image
import numpy as np

from bokeh.plotting import figure, curdoc, vplot, hplot
from bokeh.models import Button
from bokeh.client import push_session, pull_session

import dataset.mcam1
import dataset.mcam_image.local
import utils.image
import learner.mcam1


# T = time.time()
# os.mkdir('data/labeled_%d' % T)


def gen_examples():
    list_o_names = dataset.mcam_image.local.McamImage.all_names
    random.shuffle(list_o_names)

    for name in list_o_names:
        c = random.randint(55, 95)
        img = dataset.mcam_image.local.McamImage(name, c)
        slices, losses = img.image_slices_with_losses(window_size=100, stride=100, margin=10)
        slices_and_losses = zip(slices, losses)
        random.shuffle(slices_and_losses)
        for sl, l in slices_and_losses[:1]:
            yield sl, l, c

examples = gen_examples()


def plot_slice(sl, title=None):
    rgba = utils.image.arr_to_rgba(sl)
    p = figure(x_range=(0,1), y_range=(0,1), title=title)
    p.image_rgba(
        image=[rgba],
        x=[0],
        y=[0],
        dw=[1],
        dh=[1]
    )
    return p


# i = 0
# def record(sl, l, c, lbl):
#     global i
#     i += 1
#     #np.save('data/labeled_%d/%d.npy' % (T, i), sl)
#     with open('data/labeled_%d/metadata.txt' % T, 'a+') as md:
#         md.write('%d %f %d %s\n' % (i, l, c, lbl))

this_model = learner.mcam1.McamLearner(savefile='saved_1462309125_10009')
for i in range(16):
    sl, l, c = next(examples)
    [[p, _]] = this_model([sl])
    img = Image.fromarray(utils.image.arr_to_img(sl))
    img.save('test_images/%d_%d_%f_%f.png' % (i, c, l, p), 'PNG')

# plots = []
# for _ in range(7):
#     print _
#     sl, l, c = next(examples)
    #record(sl, l, c, '')
    #p = plot_slice(sl, title=str(i))
    #plots.append(p)


# session = push_session(curdoc())
# curdoc().add_root(vplot(*plots))
# session.show()

