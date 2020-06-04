# The project

The purpose is to build an algorithm to automatically identify whether a patient is suffering from breast cancer or not by looking at biopsy images. It's a binary classification problem.

<p align="center">
  <img src="https://github.com/Anthime-Biau/Image-Recognition-Project/blob/master/images/image1.PNG?raw=true" alt="Image of Tissues"/>
</p>

# Data

The dataset is composed of 300 images of tissues. 
For the training dataset :
We have 100 images of malign tissues and 100 images of benign tissues.
For the testing dataset :
We have 50 images of malign tissues and 50 images of benign tissues. 

It is a perfectly balanced dataset.

# Tools

I have decided to use Keras , it is a Python libraries which act like a deep learning API based on Tensorflow.

# Model

On Keras framework we can access pre-trained algorithm. Using pre-trained algorithm for an image classification task is a fast way to build a classifier with few training datas.

I have chosen to train a model with two pre-trained algorithm :
DenseNET201 and ResNET50, both are trained on ImageNET dataset.

# Method

