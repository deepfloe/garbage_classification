import numpy as np
from PIL import Image
from data.file_name import file_path

absolute_path = '/home/benjamin/PycharmProjects/garbage_classification/data/'
# this needs to be changed, so it can be executed on different machines
folder_names = ['cardboard','glass','metal','paper','plastic','trash']
def get_labelled_data(grayscale = True,rescale_factor=1):
    datasets = []
    labels = []
    for i , folder_name in enumerate(folder_names):
        f_path = file_path(grayscale, rescale_factor, folder_name)
        f= np.load(absolute_path+f_path+'.npy', allow_pickle=True)
        datasets.append(f)
        labels= labels+len(f)*[i]

    data = np.concatenate(datasets)
    return np.array(data), np.array(labels)

if __name__=='__main__':
    X, y = get_labelled_data(rescale_factor= 0.2)
    print(np.shape(X))
    print(np.shape(y))
    print(X[0,0:3,0:3])
