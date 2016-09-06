import glob
from PIL import Image
import numpy as np

def compute_features(nparray):
    blocksize = 8
    # TODO: implement this function
    for i in range(0, nparray.shape[1], blocksize):
        # compute a thing
    return 0, 0

images = glob.glob('data/*.jpg')

# columns: accept, dbb, dr
# where dbb is average pixel difference across vertical block boundaries
# and dr is average pixel difference inside blocks
# rows: values for each image
datarray = np.zeros(len(images), 4)

for i, image in enumerate(images):
    pil_img = Image.open(image)
    np_img = np.array(pil_img)[9:-1]
    dbb, dr = compute_features(np_img)
    datarray[i][1] = dbb
    datarray[i][2] = dr


