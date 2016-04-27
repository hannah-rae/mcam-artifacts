import dataset.base
import dataset.mcam1
import dataset.mcam_image.base
import dataset.mcam_image.sese_nfs
import dataset.mcam_image.local


def test_dataset(mcam_image_cls):
    ds = dataset.mcam1.McamDataSet(mcam_image_cls, b=0.5, w=0.5, window_size=28, stride=7, margin=10)
    slices, labels = ds.next(compression=5)
    (slice_, label) = (slices[0], labels[0])
    return slice_.shape == (28, 28, 3) and 0 <= label[0] <= 1 and label[1] == 1-label[0]

def main():
    print test_dataset(dataset.mcam_image.local.McamImage)

if __name__ == '__main__':
    main()