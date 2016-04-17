
import random
import numpy as np
import PIL


#### Query to generate artifact-image-list.csv
#### Retrieves images and their recovered products
# select sol, name, udr_image_id, filter_used from udr where udr_image_id in 
# (select udr_image_id
#    from udr
#        where type_of_product = 'RecoveredProduct'
#        and (instrument = 'ML' or instrument = 'MR')
#        and udr.name not like '%Partial%'
#        and udr.udr_image_id is NOT NULL
# order by udr_image_id)
# and type_of_product != 'Thumbnail'
# and sol != 1000
# and width = 168 and height = 150
# and name not like '%Partial%' 
# order by udr_image_id

#### Query to generate video-udrs.csv
# select distinct udr_image_id from udr where type_of_product = 'Video'


class DataSet(object):

    DEFAULT_WINDOW_SIZE = 18

    def __init__(self, images, window_size=DEFAULT_WINDOW_SIZE):
        self.images = images
        self.window_size = window_size

    def __iter__(self):
        return self

    def next(self):
        image = self.images.pop()
        uncompressed_slices = self.slice_image(image.uncompressed_data())
        compressed_slices = self.slice_image(image.compressed_data())
        next_batch = [
            (c_sl, self.compute_difference(u_sl, c_sl))
            for u_sl, c_sl in zip(uncompressed_slices, compressed_slices)
        ]
        return next_batch

    def slice_image(self, image):
        (rows, cols, chans) = image.shape
        s = self.window_size
        slices = [image[r:r+s, c:c+s] for r in range(rows-s+1) for c in range(cols-s+1)]
        return slices

    def compute_difference(self, image1, image2):
        return np.sum(np.square(image1.flatten() - image2.flatten()))


class McamImage(object):

    DEFAULT_QUALITY = 50

    def __init__(self, name, sol, udr):
        self.name = name
        self.sol = sol
        self.udr = udr
        self.quality = quality
        self._compressed_imgs = {}

    @property
    def instrument(self):
        if 'McamL' in name:
            return 'ML'
        elif 'McamR' in name:
            return 'MR'

    def uncompressed_filename(self):
        return '/molokini_raid/MSL/data/surface/processed/images/web/full/SURFACE/%s/%s/%s.png' % (
            self.instrument,
            str(self.sol).zfill(4),
            self.name.strip('\"')
        )
    
    def compressed_filename(self, quality=DEFAULT_QUALITY):
        return '/home/hannah/data/mcam-artifacts-v1/%s_compressed%d.jpg' % (
            self.name.strip('\"'),
            quality
        )

    def uncompressed_img(self):
        if not hasattr(self, '_uncompressed_img'):
            self._uncompressed_img = PIL.Image.open(self.uncompressed_filename())
        return self._uncompressed_img

    def compressed_img(self, quality=DEFAULT_QUALITY):
        if quality not in self._compressed_imgs:
            self.uncompressed_img.save(self.compressed_filename(quality=quality), 'JPEG', quality=quality)
            self._compressed_imgs[quality] = PIL.Image.open(self.compressed_filename(quality=quality))
        return self._compressed_imgs[quality]

    def uncompressed_data(self):
        return np.array(self.uncompressed_img)

    def compressed_data(self, quality=DEFAULT_QUALITY):
        return np.array(self.compressed_data(quality=quality))


def get_datasets():

    csv_image_rows = [
        (sol, name, udr, filter_)
        for (sol, name, udr, filter_) in file('artifact-image-list.csv')
        if name.startswith('McamLImage') or name.startswith('McamRImage')
    ]

    csv_recovered_rows = [
        (sol, name, udr, filter_)
        for (sol, name, udr, filter_) in file('artifact-image-list.csv')
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

    random.shuffle(recovered_mcam_images)
    random.shuffle(recovered_mcam_images)
    random.shuffle(recovered_mcam_images)
    random.shuffle(recovered_mcam_images)
    random.shuffle(recovered_mcam_images)

    def split(xs, r):
        i = int(len(xs)*r)
        return xs[:i], xs[i:]

    training_images, test_images = split(recovered_mcam_images, 0.9)

    return map(DataSet, training_images), map(DataSet, test_images)

