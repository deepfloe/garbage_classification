import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob, os

folder_names = ['cardboard','glass','metal','paper','plastic','trash']
def create_npy_files(grayscale_conversion = True):
    for i, folder_name in enumerate(folder_names):
        images_as_arrays = []
        for filename in glob.iglob('../raw_jpg/'+folder_name+'/**', recursive=False):
            if os.path.isfile(filename): # filter dirs
                if grayscale_conversion:
                   image =Image.open(filename).convert('L')
                else:
                    image = Image.open(filename)
                images_as_arrays.append(np.asarray(image))
        if grayscale_conversion:
            target_dir = 'grayscale_'+folder_name
        else:
            target_dir = 'rgb_'+folder_name

        np.save(target_dir,images_as_arrays)

if __name__=='__main__':
    #create_npy_files(grayscale_conversion = True)
