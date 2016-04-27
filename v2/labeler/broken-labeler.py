
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


T = time.time()
os.mkdir('data/labeled_%d' % T)


def gen_examples():
    dset = dataset.mcam1.McamDataSet(
        dataset.mcam_image.local.McamImage,
        b = 0.5,
        w = 0.5,
        window_size = 48,
        stride = 48*4
    )
    while True:
        c = random.randint(60 , 95)
        slices, _ = dset.next(compression=c)
        for sl in slices:
            yield sl, c

examples = gen_examples()


def plot_slice(sl):
    rgba = utils.image.arr_to_rgba(sl)
    p = figure(x_range=(0,1), y_range=(0,1))
    p.image_rgba(
        image=[rgba],
        x=[0],
        y=[0],
        dw=[1],
        dh=[1]
    )
    return p


i = 0
def record(sl, c, lbl):
    global i
    i += 1
    np.save('data/labeled_%d/%d.npy' % (T, i), sl)
    with open('data/labeled_%d/metadata.txt' % T, 'a+') as md:
        md.write('%d %d %s\n' % (i, c, lbl))


def do_step(document):
    document.clear()

    sl, c = next(examples)

    p = plot_slice(sl)

    b_A = Button(label="Accept")
    b_R = Button(label="Reject")

    b_A.on_click(functools.partial(callback, sl, c, 'accept'))
    b_R.on_click(functools.partial(callback, sl, c, 'reject'))

    plot = vplot(p, (hplot(b_A, b_R)))

    document.add_root(plot)


def callback(sl, c, lbl):
    print '''
    Entered callback
    '''
    session = pull_session(session_id='labeler')
    document = session.document
    document.clear()

    # record(sl, c, lbl)
    # sl, c = next(examples)
    # widget(sl, c)
    print '''
    Returning
    '''


def main():
    session = push_session(curdoc(), session_id='labeler')
    document = session.document
    print document
    do_step(document)
    session.show()
    session.loop_until_closed()


main()


