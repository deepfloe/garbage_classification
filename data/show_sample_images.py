from PIL import Image
from file_name import file_path
import numpy as np
import matplotlib.pyplot as plt

absolute_path = '/home/benjamin/PycharmProjects/garbage_classification/data/'

def show_sample_images(grayscale, rescale_factor, label):
    f_path = file_path(grayscale, rescale_factor, label)
    f = np.load(absolute_path + f_path + '.npy', allow_pickle=True)
    image = Image.fromarray(f[0])
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    show_sample_images(True, 0.2 , 'paper')