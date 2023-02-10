import splitfolders

if __name__ == '__main__':
    # ToDo: check whether the split has been executed already
    splitfolders.ratio('/home/benjamin/PycharmProjects/garbage_classification/raw_jpg/', ratio=(0.8,0.1,0.1), output='/home/benjamin/PycharmProjects/garbage_classification/raw_jpg/')