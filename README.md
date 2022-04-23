# Concrete_Cracks_Image_Classification
 
## 1. Summary
<p> This project aims to create a convolutional neural network model that can detect cracks on concrete with high accuracy.<br> 
This problem is a binary classfication problem (no cracks/negative and cracks/positive).<br>
The model is trained with a dataset contains of 40000 images consisted of 20000 images of concrete in good condition and 20000 images of concrete with cracks.<br> 

 The source of the dataset can be found [here](https://data.mendeley.com/datasets/5y9wdsg2zt/2).</p> 

## 2. IDE and Framework
<p>This project is done using Spyder as the main IDE. The main frameworks used in this project are Matplotlib, Numpy and Tensorflow Keras.</p>

## 3. Methodology

<p>This methodology is referred to a documentation on the official TensorFlow website (https://www.tensorflow.org/tutorials/images/transfer_learning).</p>

### 3.1 Data Pipeline
<p>The image data are loaded along with their corresponding labels. Firstly, the data is splitted into train-validation set with a ratio of 70:30. The validation data is then further split into two group to obtain some test data, with a ratio of 80:20. The overall train-validation-test split ratio is 70:24:6. No data augmentation is applied as the data size and variation are already sufficient.</p>

### 3.2 Model Pipeline
<p>The input layer is designed to receive coloured images with a dimension of 160x160. The full shape will be (160,160,3).

Transfer learning is applied for building the deep learning model of this project. Firstly, a preprocessing layer is created that will change the pixel values of input images to a range of -1 to 1. This layer serves as the feature scaler and it is also a requirement for the transfer learning model to output the correct signals.

For feature extractor, a pretrained model of MobileNet v2 is used. The model is readily available within TensorFlow Keras package, with ImageNet pretrained parameters. It is also frozen hence will not update during model training.

A global average pooling and dense layer are used as the classifier to output softmax signals. The softmax signals are used to identify the predicted class.

The simplified illustration of the model is shown below.</p>

## 4. Results
<p>The loss and accuracy are shown below.</p>
 
![result2](https://user-images.githubusercontent.com/72061179/164844902-a055a590-9c67-4027-83b8-0e9aab672fd4.png)
 
<p>Some predictions are also been made with the model and compared with the actual results.
</p>

![result](https://user-images.githubusercontent.com/72061179/164838410-0ccedcf5-5b97-429a-8f54-ccb7dd5bfa31.png)
