
import os

from PIL import Image
import numpy as np


class BaseMcamImage(object):

    tmp_dir = NotImplemented

    def __init__(self, name, compression=None):
        self.name = name
        self.compression = compression

    @property
    def _raw_image_path(self):
        raise NotImplementedError

    @property
    def raw_image(self):
        if not hasattr(self, '_raw_image'):
            raw_PIL_image = Image.open(self._raw_image_path)
            self._raw_image = np.array(raw_PIL_image) / 256.
        return self._raw_image

    @property
    def image(self):
        if not hasattr(self, '_image'):
            if self.compression:
                raw_PIL_image = Image.open(self._raw_image_path)
                image_path = self.tmp_dir + self.name + '_compressed%d.jpg' % self.compression
                raw_PIL_image.save(image_path, 'JPEG', quality=self.compression)
                PIL_image = Image.open(image_path)
                os.remove(image_path)
                self._image = np.array(PIL_image) / 256.
            else:
                self._image = self.raw_image
        return self._image

    def image_slices_with_losses(self, window_size, stride, margin):

        def slice_image(image):
            (rows, cols, chans) = image.shape
            return [
                image[r:r+window_size, c:c+window_size]
                for r in range(margin, rows - margin - window_size + 1, stride)
                for c in range(margin, cols - margin - window_size + 1, stride)
            ]

        slices = slice_image(self.image)

        raw_slices = slice_image(self.raw_image)

        losses = [
            self.compute_loss(sl, r_sl)
            for (sl, r_sl) in zip(slices, raw_slices)
        ]

        # for i, s in enumerate(slices[:10]):
        #     print s.shape
        #     s_pil = Image.fromarray(s, 'RGB')
        #     s_pil.save(self.tmp_dir + self.name + '_%dcompressed%d.jpg' % (i, self.compression), 'PNG')

        return slices, losses

    def compute_loss(self, image1, image2):
        # return float(np.sqrt(np.sum(np.square(image1.flatten() - image2.flatten()))))
        return float(np.mean(np.abs(image1 - image2)))













