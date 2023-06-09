# Garbage Classification
Testing different machine learning models to classify garbage images [from this kaggle dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification).
As often the case in image classification tasks, one key challenge is the small dataset. On the other hand, no pre-processing is required on the dataset. The images have the same resolution, the photos of the objects are taken against a contrasting background.

## The dataset
~2500 RGB images of garbage items of size 512 x 384. They are labelled into six categories, 
with the following counts cardboard: 403, glass: 501, metal: 410, paper: 594, plastic 482, trash: 137. So the dataset is fairly balanced, apart from the underrepresented trash category.
The dataset is split into training, test, val with the ratio -(0.5,0.25,0.25).

<img src="https://user-images.githubusercontent.com/53785628/225296951-312184de-cfe3-4ebe-a96e-3be6b2fe2a0d.jpg"  width="30%" height="30%"><img src="https://user-images.githubusercontent.com/53785628/225297043-1be96046-ec2e-4700-9db3-560a1308abb0.jpg"  width="30%" height="30%"><img src="https://user-images.githubusercontent.com/53785628/225297095-c2de894f-f825-4305-9aa5-87e9c5f27541.jpg"  width="30%" height="30%">
## Data augmentation
A common technique in image classification is data augmentation. The model becomes more robust by applying transformations to the input during training, in our case a random horizontal flip, zoom or rotation:

<img src="https://user-images.githubusercontent.com/53785628/225303330-6e61c6c5-9afc-4aa0-a63d-86d76ab10875.png"  width="40%" height="40%">

## The models
- **Convnet from scratch with data augmentation**: a simple convnet model, entirely trained from scratch with three convolutional+maxpooling layers [kaggle

- **VGG16 without data augmentation**: uses the vgg16 convbase to preprocess data and then trains a small model on top. The advantage is that this is very fast, as applying the conv base only happens once.
The disadvantage is that it cannot be combined with data augmentation.

- **VGG16 with data augmentation**: Here the conv base is a non-trainable part of the whole model, which is applied after the data augmentation layer. Training is significantly slower thann in the second model.

## How to use 

1. download the dataset from [kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification), extract into a folder called "data" in the root repo
2. run the file `train_test_val_splitfolders.py` which creates the subfolders called "train/", "test/" and "val/" in data
3. On a **local machine**, to train the three different models, execute `train_convnet_from_scratch()`, `train_vgg()`, `train_vgg_augment()` in the file `training.py`. If on **google colab**, execute 1. and 2., upload the datafolder to google drive and run the whole script `full_google_colab_script.pynb`

## Results

### Convnet from scratch
We train for 50 epochs and achieve a **test accuracy of 58%**. Overfitting starts around epoch 25

![from_scratch](https://user-images.githubusercontent.com/53785628/225329770-a758d5ac-ab94-417c-acf8-075de4f87f75.png)

### VGG16 without data augmentation
We train for 25 epochs and achieve a **test accuracy of 85%**. The validation loss is relatively constant from epoch 5 onwards.
![vgg](https://user-images.githubusercontent.com/53785628/225336625-e5f89fd7-9881-4eb1-8bde-ac62cc6ca7f5.png)

### VGG16 with data augmentations
We train for 15 epochs and achieve a **test accuracy of 82%**. The val loss is fairly stable from epoch 3 onwards.
![vgg_augment](https://user-images.githubusercontent.com/53785628/225597208-3fabadfa-048c-4ece-94d1-c0c21eb7e21d.png)

## Questions and issues
- Why does data augmentation not increase the test accuracy in the vgg model?
- Using the function `vgg_preprocessing_datasets(train_dataset, val_dataset, test_dataset)` in `pre_processing.py` is much slower in training than `vgg_preprocessing(train_dataset, val_dataset, test_dataset)`. There seems to be an issue with `tensorflow.Data.dataset.map` method.

## Further directions
- The accuracy could posibly be improved by fine tuning the combined vgg model with a small learning rate or by using a state-of-the art image classification model as a base for transfer learning, for example inception.
- As the dataset is quite small, k-cross-validation could be useful

## References
The main reference for this project is chapter 8 of 
François Chollet - Deep Learning with Python- Manning Publication 2021



