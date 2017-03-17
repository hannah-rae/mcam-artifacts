import random
import time
import threading
import os
import functools

from PIL import Image
import numpy as np

import dataset.mcam1
import dataset.mcam_image.local
import utils.image
import learner.mcam1


img = dataset.mcam_image.local.McamImage('McamRRecoveredProduct_0490433279-04054-1', 75)
slices, losses = img.image_slices_with_losses(window_size=160, stride=160, margin=10)
slices_and_losses = zip(slices, losses)

this_model = learner.mcam1.McamLearner(window_size=160, savefile='saved_sessions/saved_1488556921_10535.meta')
print(len(slices_and_losses))
i = 0
for sl, l in slices_and_losses[:34]:
    [[p, _]] = this_model([sl])
    img = Image.fromarray(utils.image.arr_to_img(sl))
    img.save('test_images/%d_%d_%f_%f.png' % (i, 85, l, p), 'PNG')
    i += 1