import matplotlib.pyplot as plt
import os

def explore_training_set():
    train_dir = '/home/benjamin/PycharmProjects/garbage_classification/raw_jpg/train'
    counts = {}
    labels = os.listdir(train_dir)
    for l in labels:
        counts[l] = len(os.listdir(train_dir+'/'+l))


    print(counts)
    test_size = sum(counts.values())
    print('total number of elements:', test_size)
    print('baseline:',max(counts.values())/test_size)


    plt.pie(counts.values(), labels = labels)
    plt.show()
