
import random
import os
import threading
import Queue

import numpy as np
from PIL import Image


#### Query to generate artifact-image-list.csv
#### Retrieves images and their recovered products
# select sol, name, udr_image_id, filter_used, comp_quality from udr
#   where udr_image_id in
#       (select udr_image_id from udr
#          where type_of_product = 'RecoveredProduct'
#          and (instrument = 'ML' or instrument = 'MR')
#          and udr.name not like '%Partial%'
#          and udr.udr_image_id is NOT NULL
#          and comp_quality = 0
#        order by udr_image_id)
#   and type_of_product != 'Thumbnail'
#   and sol != 1000
#   and width = 168 and height = 150
#   and name not like '%Partial%'
# order by udr_image_id

#### Query to generate video-udrs.csv
# select distinct udr_image_id from udr where type_of_product = 'Video'


WINDOW_SIZE = 28

STRIDE = WINDOW_SIZE * 4

# QUALITY_BOUNDS = (5, 95)
QUALITY_SAMPLES = [20,95]
COMPRESSED = 1
UNCOMPRESSED = 0

IMGS_PER_BATCH = len(QUALITY_SAMPLES)

TRAIN_TEST_SPLIT = 0.9

NUM_WORKERS = 2

class DataSet(object):

    def __init__(self, images, N=1, schedule=None):
        self.lock = threading.Lock()
        self.images = images * N
        self.window_size = WINDOW_SIZE

    @property
    def batch_size(self):
        if not hasattr(self, '_batch_size'):
            self.lock.acquire()
            frags_per_img = len(self.frag_image(self.images[0].uncompressed_data()))
            self._batch_size = frags_per_img * IMGS_PER_BATCH
            self.lock.release()
        return self._batch_size

    def __iter__(self):
        return self

    def next(self):
        next_batch = []
        for i, q in enumerate(QUALITY_SAMPLES):
            self.lock.acquire()
            if not self.images:
                raise StopIteration
            else:
                image = self.images.pop()
            self.lock.release()
            compressed_frags = self.frag_image(image.compressed_data(quality=q))
            next_batch += [(c_sl, i) for c_sl in compressed_frags]
        random.shuffle(next_batch)
        def unzip(xys):
            return tuple(map(list, zip(*xys)))
        return unzip(next_batch)

    def frag_image(self, image):
        (rows, cols, chans) = image.shape
        s = self.window_size
        frags = [
            image[r:r+s, c:c+s]
            for r in range(0, rows-s+1, STRIDE) for c in range(0, cols-s+1, STRIDE)
            if (10 < r < rows-10-s) and (10 < c < cols-10-s)  # exclude edges
        ]
        return frags

    def compute_corruption(self, image1, image2):
        return float(np.sqrt(np.sum(np.square(image1.flatten() - image2.flatten()))))


class McamImage(object):

    DEFAULT_QUALITY = 40

    def __init__(self, name, sol, udr):
        self.name = name
        self.sol = sol
        self.udr = udr
        self._compressed_imgs = {}

    @property
    def instrument(self):
        if 'McamL' in self.name:
            return 'ML'
        elif 'McamR' in self.name:
            return 'MR'

    def uncompressed_filename(self):
        return '/molokini_raid/MSL/data/surface/processed/images/web/full/SURFACE/%s/sol%s/%s.png' % (
            self.instrument,
            str(self.sol).zfill(4),
            self.name.strip('\"')
        )

    def compressed_filename(self, quality=DEFAULT_QUALITY):
        return '/home/hannah/data/mcam-artifacts-tmp/%s_compressed%d.jpg' % (
            self.name.strip('\"'),
            quality
        )

    def uncompressed_img(self):
        if not hasattr(self, '_uncompressed_img'):
            self._uncompressed_img = Image.open(self.uncompressed_filename())
        return self._uncompressed_img

    def compressed_img(self, quality=DEFAULT_QUALITY):
        if quality not in self._compressed_imgs:
            self.uncompressed_img().save(self.compressed_filename(quality=quality), 'JPEG', quality=quality)
            self._compressed_imgs[quality] = Image.open(self.compressed_filename(quality=quality))
            os.remove(self.compressed_filename(quality=quality))
        return self._compressed_imgs[quality]

    def uncompressed_data(self):
        return np.array(self.uncompressed_img()) / 256.

    def compressed_data(self, quality=DEFAULT_QUALITY):
        return np.array(self.compressed_img(quality=quality)) / 256.


def get_datasets():

    csv_rows = [
        (lambda w: (int(w[0]), w[1], int(w[2]), (w[3].strip())))(line.split(','))
        for line in file('artifact-image-list.csv')
    ]

    csv_image_rows = [
        (sol, name, udr, filter_) for (sol, name, udr, filter_) in csv_rows
        if name.startswith('McamLImage') or name.startswith('McamRImage')
    ]

    csv_recovered_rows = [
        (sol, name, udr, filter_) for (sol, name, udr, filter_) in csv_rows
        if name.startswith('McamLRecoveredProduct') or name.startswith('McamRRecoveredProduct')
    ]

    udr_blacklist = [row.rstrip('\n') for row in file('video-udrs.csv')]

    def maybe_get_recovered_name_from_udr(udr):
        recovered_names = [
            name for (sol, name, udr_, filter_) in csv_recovered_rows
            if udr_ == udr
        ]
        if recovered_names:
            return recovered_names[0]
        else:
            return None

    def maybe_make_McamImage(maybe_name, sol, udr):
        if maybe_name:
            return McamImage(maybe_name, sol, udr)
        else:
            return None

    recovered_mcam_images = filter(lambda x: x is not None, [
        maybe_make_McamImage(maybe_get_recovered_name_from_udr(udr), sol, udr)
        for (sol, name, udr, filter_) in csv_image_rows
        if (filter_ == '0') and (udr not in udr_blacklist)
    ])

    random.seed(8008135)
    random.shuffle(recovered_mcam_images)

    def make_datasets(imgs, ratio):
        i = int(ratio*len(imgs))
        return DataSet(imgs[:i]), DataSet(imgs[i:])

    training_dataset, test_dataset = make_datasets(recovered_mcam_images, TRAIN_TEST_SPLIT)

    print 'Images found: %d' % len(recovered_mcam_images)
    print 'Batch size: %d' % training_dataset.batch_size

    return training_dataset, test_dataset


def make_dataqueue(dataset):
    dataqueue = Queue.Queue(maxsize=4*NUM_WORKERS)
    def worker():
        for batch in dataset:
            dataqueue.put(batch)
        dataqueue.put(None)  # sentinel
    for _ in xrange(NUM_WORKERS):
        worker_thread = threading.Thread(target=worker)
        worker_thread.start()
    return dataqueue


if __name__ == '__main__':
    training_dataset, test_dataset = get_datasets()
    filenames = [img.uncompressed_filename() for img in training_dataset.images] + [img.uncompressed_filename() for img in test_dataset.images]
    for filename in filenames:
        print filename




