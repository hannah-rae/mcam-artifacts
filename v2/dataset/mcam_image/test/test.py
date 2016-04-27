
import dataset.mcam_image.base
import dataset.mcam_image.sese_nfs
import dataset.mcam_image.local


def test_all_names(mcam_image_cls):
    return 'McamRRecoveredProduct_0496482327-26530-1' in mcam_image_cls.all_names


def test_image_slices_with_losses(mcam_image_cls):
    image = mcam_image_cls('McamRRecoveredProduct_0496482327-26530-1', compression=20)
    slices, losses = image.image_slices_with_losses(window_size=100, stride=200, margin=10)
    (slice_, loss) = (slices[0], losses[0])

    return slice_.shape == (28, 28, 3) and 0 <= loss <= 1


def main():
    print test_all_names(dataset.mcam_image.local.McamImage)
    print test_image_slices_with_losses(dataset.mcam_image.local.McamImage)

    print test_all_names(dataset.mcam_image.sese_nfs.McamImage)
    # print test_image_slices_with_losses(dataset.mcam_image.sese_nfs.McamImage)


if __name__ == '__main__':
    main()
