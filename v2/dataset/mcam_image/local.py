import glob

import dataset.mcam_image.base

class McamImage(dataset.mcam_image.base.BaseMcamImage):

    data_dir = '/Users/hannahrae/data/mcam-data/'
    save_dir = data_dir + 'saved/'
    tmp_dir = data_dir + 'tmp/'

    all_names = [
        fn.split('/')[-1].rstrip('.png')
        for fn in glob.glob(data_dir + '*.png')
    ]

    @property
    def _raw_image_path(self):
        return self.data_dir + self.name + '.png'