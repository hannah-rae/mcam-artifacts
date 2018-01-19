import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image

mcam_dir = '/Users/hannahrae/data/mcam-data'
live_dir = '/Users/hannahrae/data/livedatabase/jpeg'

def load_mcam_images():
    num_images = len(glob(mcam_dir + '/*.png'))
    # Create an empty array to hold all the images
    dataset = np.zeros((num_images, 1200, 1344, 3))
    for i, img_fn in enumerate(glob(mcam_dir + '/*.png')):
        image = Image.open(img_fn).convert('RGB')
        dataset[i] = np.array(image)
    return dataset

def load_live_images():
    num_images = len(glob(live_dir + '/*.bmp'))
    # Create an empty array to hold all the images
    dataset = np.zeros((num_images, 768, 720, 3))
    for i, img_fn in enumerate(glob(live_dir + '/*.bmp')):
        image = Image.open(img_fn)
        # TODO: pad images with 0 to max (768, 720) instead of resizing to max?
        image = image.resize((720, 768))
        dataset[i] = np.array(image)
    return dataset

def color_hist():
    # Load image data for both datasets
    mastcam = load_mcam_images()
    live = load_live_images()

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    # Red
    axs[0].hist(mastcam[:,:,:,0].flatten(), bins=range(256), normed=True, alpha=0.5, label='Mastcam')
    axs[0].hist(live[:,:,:,0].flatten(), bins=range(256), normed=True, alpha=0.5, label='LIVE')
    axs[0].set_xticks(range(0, 256, 17))
    axs[0].set_ylabel("Normalized Number of Pixels")
    axs[0].set_xlabel("Digital Number (DN)")
    axs[0].set_title("Histogram Comparison of Red Channel")
    axs[0].legend(loc='upper right')
    # Green
    axs[1].hist(mastcam[:,:,:,1].flatten(), bins=range(256), normed=True, alpha=0.5, label='Mastcam')
    axs[1].hist(live[:,:,:,1].flatten(), bins=range(256), normed=True, alpha=0.5, label='LIVE')
    axs[1].set_xticks(range(0, 256, 17))
    axs[1].set_ylabel("Normalized Number of Pixels")
    axs[1].set_xlabel("Digital Number (DN)")
    axs[1].set_title("Histogram Comparison of Green Channel")
    axs[1].legend(loc='upper right')
    # Blue
    axs[2].hist(mastcam[:,:,:,2].flatten(), bins=range(256), normed=True, alpha=0.5, label='Mastcam')
    axs[2].hist(live[:,:,:,2].flatten(), bins=range(256), normed=True, alpha=0.5, label='LIVE')
    axs[2].set_xticks(range(0, 256, 17))
    axs[2].set_ylabel("Normalized Number of Pixels")
    axs[2].set_xlabel("Digital Number (DN)")
    axs[2].set_title("Histogram Comparison of Blue Channel")
    axs[2].legend(loc='upper right')
    plt.show()

def freq_hist():
    # Load image data for both datasets
    mastcam = load_mcam_images()
    live = load_live_images()

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def rgb2freq(im):
        im = rgb2gray(im)
        f = np.fft.fft2(im)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        return magnitude_spectrum

    ## Plot entire dataset
    mastcam_f = np.array([rgb2freq(im) for im in mastcam])
    live_f = np.array([rgb2freq(im) for im in live])

    plt.hist(mastcam_f.ravel(), alpha=0.5, label='Mastcam', bins=300, normed=True)
    plt.hist(live_f.ravel(), alpha=0.5, label='LIVE', bins=300, normed=True)
    plt.title('Histogram of Magnitude Spectra')
    plt.xlabel('Frequency')
    plt.ylabel('Occurrence of Frequencies in Dataset\n Converted to Spatial Domain')
    plt.legend(loc='upper right')
    plt.show()

    ## Plot single image examples
    # img = live[0][:,:,0]
    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift))

    # img2 = mastcam[0][:,:,0]
    # f2 = np.fft.fft2(img2)
    # fshift2 = np.fft.fftshift(f2)
    # magnitude_spectrum2 = 20*np.log(np.abs(fshift2))

    # fig, ax = plt.subplots(nrows=2, ncols=3)
    # ax[0,0].hist(magnitude_spectrum.ravel(), bins=100)
    # ax[0,0].set_title('Histogram of Magnitude Spectrum')
    # ax[0,1].imshow(magnitude_spectrum, interpolation="none")
    # ax[0,1].set_title('Magnitude Spectrum')
    # ax[0,2].imshow(img, interpolation="none")
    # ax[1,0].hist(magnitude_spectrum2.ravel(), bins=100)
    # ax[1,0].set_title('Histogram of Magnitude Spectrum')
    # ax[1,1].imshow(magnitude_spectrum2, interpolation="none")
    # ax[1,1].set_title('Magnitude Spectrum')
    # ax[1,2].imshow(img2, interpolation="none")
    # plt.show()

def mean_image():
    return

def main():
    # mean_image()
    #color_hist()
    freq_hist()

if __name__ == '__main__':
    main()