import numpy as np
from PIL import Image
folder_names = ['cardboard','glass','metal','paper','plastic','trash']
def get_labelled_data(grayscale = True):
    datasets = []
    labels = []
    for i , folder_name in enumerate(folder_names):
        if grayscale:
            filename = 'grayscale_'+folder_name+'.npy'
        else:
            filename = 'rgb_' + folder_name + '.npy'
        f= np.load(filename, allow_pickle=True)
        datasets.append(f)
        labels= labels+len(f)*[i]

    data = np.concatenate(datasets)
    return np.array(data), np.array(labels)

if __name__=='__main__':
    X, y = get_labelled_data()

