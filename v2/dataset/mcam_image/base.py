
import os
import math

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys


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

    def joint_hist(self, im1, im2):
        joint_hist = np.zeros((256, 256))
        joint = zip(im1, im2)
        for (i,j) in joint:
            joint_hist[i][j] += 1
        return joint_hist


    def compute_loss(self, image1, image2):
        # return float(np.sqrt(np.sum(np.square(image1.flatten() - image2.flatten()))))
        # return float(np.mean(np.abs(image1 - image2)))

        # r1 = np.asarray(image1[:,:,0])
        # r1 = np.multiply(r1, 256)
        # g1 = np.asarray(image1[:,:,1])
        # g1 = np.multiply(g1, 256)
        # b1 = np.asarray(image1[:,:,2])
        # b1 = np.multiply(b1, 256)

        # hr1, h_bins = np.histogram(r1, bins=256, normed=True)
        # hg1, h_bins = np.histogram(g1, bins=256, normed=True)
        # hb1, h_bins = np.histogram(b1, bins=256, normed=True)
        # hist1 = np.array([hr1, hg1, hb1]).ravel()

        # r2 = np.asarray(image2[:,:,0])
        # r2 = np.multiply(r2, 256)
        # g2 = np.asarray(image2[:,:,1])
        # g2 = np.multiply(g2, 256)
        # b2 = np.asarray(image2[:,:,2])
        # b2 = np.multiply(b2, 256)

        # hr2, h_bins = np.histogram(r2, bins=256, normed=True)
        # hg2, h_bins = np.histogram(g2, bins=256, normed=True)
        # hb2, h_bins = np.histogram(b2, bins=256, normed=True)
        # hist2 = np.array([hr2, hg2, hb2]).ravel()

        # diff = hist2 - hist1
        # squares = np.multiply(diff, diff)
        # distance = np.sqrt(sum(squares))

        # return distance

        img1 = image1.flatten()
        img1 = [int(px*256) for px in img1]
        img2 = image2.flatten()
        img2 = [int(px*256) for px in img2]
        joint_histogram = self.joint_hist(img1, img2)
        prob_without_zeros = []
        for row in joint_histogram:
            for val in row:
                if val != 0: # remove 0 values for log operation
                    val = float(val)/len(img1) # normalize
                    prob_without_zeros.append(val)
        logged = [math.log(x, 2) for x in prob_without_zeros]
        joint_entropy = -sum(np.multiply(prob_without_zeros, logged))
        return joint_entropy

        # V1
        # pixel_diffs = np.abs(image1 - image2)
        # pixel_diffs.sort() # sort ascending
        # top_diff = float(np.mean(pixel_diffs[11:])) # take the average of the top 10 highest differences
        # print top_diff
        # return top_diff












