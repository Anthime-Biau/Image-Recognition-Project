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

For the two selected models :
I used DenseNET201 and ResNET50 pre trained weights which is already trained in the Imagenet competition. The learning rate chosen to be 0.0001.

On top of them I used a globalaberagepooling layer followed by 50% dropouts to reduce over-fitting.

I used batch normalization and a dense layer with 2 neurons for 2 output classes (benign and malign) with softmaw as the activation function.

I have used Adam as the optimizer and binary-cross-entropy as the loss function.

```

def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    return model

DenseNet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

ResNet = ResNet50(
    weights= 'imagenet',
    include_top=False,
    input_shape= (224, 224, 3)
)
```

# Method

As I do not have a lot of images in my training set, and not a huge computational power I decided to use an ImageDataGenerator. It appears as a good way to generate new images at each epoch and avoid to store them directly in memory.

```
train_generator = ImageDataGenerator(
        zoom_range=2,  # set range for random zoom
        rotation_range = 90,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

```

Before training the model, it is useful to define one or more callbacks. I have chosen ModelCheckpoint and ReduceLROnPlateau.

* ModelCheckpoint : As training requires a lot of time to achieve a good result, often many iterations are required. It is better to save a copy of the best performing model only when an epoch imporves the metrics ends.

* ReduceLROnPlateau : Reduce learning rate when a metric has stopped improving. If there is no improvement seen for a given 'patience' number of epochs, the laearning rate is reduced.

I have decided to trained the model for 50 epochs



