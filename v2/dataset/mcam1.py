import dataset.base
import utils.locked

import scipy.special
import numpy as np

class McamDataSet(dataset.base.DataSet):

    def __init__(self, image_class, b=None, w=None, window_size=None, stride=None, margin=10):
        if any(map(lambda x: x is None, [b, w, window_size, stride])):
            raise RuntimeError('McamDataSet.__init__: Missing argument')

        self.image_class = image_class
        self.image_names = utils.locked.Locked(list(image_class.all_names)*15)  # increases amount of data
        self.b = b
        self.w = w
        self.window_size = window_size
        self.stride = stride
        self.margin = margin

    def next(self, compression=None):
        if not compression:
            raise RuntimeError('McamDataSet.next: Missing argument')

        with self.image_names as image_names:
            if not image_names:
                raise StopIteration

            image = self.image_class(image_names.pop(), compression=compression)

            slices, losses = image.image_slices_with_losses(
                window_size=self.window_size,
                stride=self.stride,
                margin=self.margin
            )

            labels = [self.loss_to_label(loss, compression) for loss in losses]

            return slices, labels

    def loss_to_label(self, loss, compression):

        def predict_1d(x, theta=-1.38705672, b=14.83335514):
            return 1 / (1 + np.exp(-1 * (np.dot(theta, x)+b)))

        def predict_2d(x, theta=np.array([-1.20730291, 0.15657625]), b=0):
            return 1 / (1 + np.exp(-1 * (np.dot(theta, x)+b)))

        prob = predict_2d(np.array([loss, compression]))
        #prob = predict_1d(loss)
        #prob = scipy.special.expit(self.w*loss + self.b)
        return [prob, 1-prob]


