import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob, os
from file_name import file_path
from data.scaled_input_size import scaled_input_size
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
def create_npy_files(grayscale_conversion = True,rescale_factor=1):
    new_size = scaled_input_size(rescale_factor)
    #create a separate file for each label to decrease the file size
    for i, folder_name in enumerate(labels):
        images_as_arrays = []
        #resize, grayscale images and convert into numpy array
        for filename in glob.iglob('../raw_jpg/'+folder_name+'/**', recursive=False):
            if os.path.isfile(filename): # filter dirs
                if grayscale_conversion:
                   image =Image.open(filename).convert('L')
                else:
                    image = Image.open(filename)

                image = image.resize((new_size))
                images_as_arrays.append(np.asarray(image))
        #storing the numpy array in an npy file
        target_path = file_path(grayscale_conversion,rescale_factor,folder_name)
        np.save(target_path,images_as_arrays)

if __name__=='__main__':
    create_npy_files(grayscale_conversion = True, rescale_factor=0.2)
