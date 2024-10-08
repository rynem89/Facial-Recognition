# Import the TensorFlow library
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Load the ORL faces dataset
# Dataset Details: The ORL Database of Faces consists of 400 images from 40 different subjects.
# The images were captured at different times, under varying lighting conditions, with different 
# facial expressions (open, closed eyes, smiling, not smiling), and with or without glasses.
# All the images have a dark homogeneous background, and the subjects are positioned upright and frontal with some tolerance for side movement.
# Each image has a size of 92x112 pixels and 256 grey levels per pixel.
# Read a npz file containing the images
data = np.load('ORL_faces.npz')

# Checking the number and Shape of file and type of data
print('The data contains following files: ',data.files)

# Exploring the individual files containing the training and testing images
image_train = data['trainX']
image_test = data['testX']
label_train = data['trainY']
label_test = data['testY']
print(image_train)
print(image_train.shape)
print(type(image_train))
print(label_train)
print(label_train.shape)
print(type(label_train))
print(image_test)
print(image_test.shape)
print(type(image_test))
print(label_test)
print(label_test.shape)
print(type(label_test))

# PART 1 : Training without Transfer Learning

# Files contain images as numpy arrays and we can assume with 240 images in trainX for training , and 160 images for in textX for testing along with their corresponding labels
# in testX and testY, respectively.
# Number of classes in the dataset for training and testing
print('Number of classes of trainging dataset: ', len(np.unique(label_train)))
print('Number of classes of testing dataset: ', len(np.unique(label_test)))

# Since each image has a size of 92x112x1=10304 pixels and 256 grey levels per pixel ie, 1 channel.
# Also as there are 400 images and since 240 are for training then we can confirm rest 160 for testing.
# However the labels for training and testing suggest that there are are only 20 uniquel labels, therefore just 20 different people images and not 40 that was in the original ORL
# dataset. 

n_rows = 112
n_cols = 92
n_channels = 1

# Reshape and resizing the data content to given 
trainX =  np.reshape(image_train,newshape=(data['trainX'].shape[0],n_rows,n_cols, n_channels))
trainX = trainX/255.
testX =  np.reshape(image_test,newshape=(data['testX'].shape[0],n_rows,n_cols, n_channels))
testX = testX/255.

# Check the new shape
print('Trainging and testing data shape:',trainX.shape, testX.shape)

# Visualise train images each person image as the images are saved in sequece ie. 10 variations of each person stroed consecutively
i=0
j=0
# creating an 2D array to store this images and their lables
images = np.zeros((20,112,92,1))
plt.figure(figsize=(10,10))
for i in range(20):
    i = np.max(np.where(label_train==j)) # find location index with value corresponding to values from 0 to 19
    plt.subplot(4,5,j+1)
    plt.imshow(trainX[i], cmap='gray')
    plt.title(label_train[i])
    plt.axis('off')
    j= j+1
    images[j-1] = trainX[i]
plt.show()


# Since the testing image files contains 40% (160) of the total 400 images which typical a huge number considering the ratio of training vs testing and is not necessary. Furethermore,
# this extra images of testing can be used to train the model better by utilising half of it for validation during the training. 
# Therefore, we split the tesitng dataset by 50% for testing and validation equallly.
# Spliting the dataset for testing and validation
X_test, X_val, y_test, y_val = train_test_split(testX, label_test, test_size=0.5, random_state=42, stratify=label_test)

# Assinging new variables for training dataset
X_train = trainX
y_train = label_train

# Number of training vs validation dataset of training dataset and testing dataset
print('Training data size: ', X_train.shape[0])
print('Validation data size: ', X_val.shape[0])
print('Training data label size: ', y_train.shape[0])
print('Validation data label size: ', y_val.shape[0])
print('Testing data size: ', X_test.shape[0])
print('Testing data label size: ', y_test.shape[0])


# Define the CNN model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same',input_shape=(112, 92, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),     # Dropout layer to reduce overfitting
    tf.keras.layers.Dense(20, activation='softmax')])


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.save_weights('best_model.h5')
model.summary()

# Training the model 
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data= (X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[   # callback to save best accuracy model weights and reduce learning rate if loss remains same
        tf.keras.callbacks.ModelCheckpoint(filepath="best_model.h5",monitor="val_accuracy",verbose=1,save_best_only=True,save_weights_only=True,
        mode="auto",save_freq="epoch",initial_value_threshold=None,),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=20,min_lr=0.0001,factor=0.1)])

# Save the model
model.save('My_model')

# Load the model weights with best validation accuracy
model.load_weights('best_model.h5')
# Evaluate the model on the test data
results = model.evaluate(x=X_test, y=y_test, verbose = 0)
print('Test Loss: {:.5f}'.format(results[0]))
print('Test Accuracy: {:.2f}%'.format(results[1] * 100))

# Prdicting 10 random image classes using the trained model
j=0
k=1
plt.figure(figsize=(7,10))
for m in range(10):
    rand = np.random.randint(0, len(X_test))
    test = X_test[rand]
    print('Image to be Predicted with label: {}'.format(y_test[rand]))
    # Converting the image dimension for model input compatibility
    test = np.expand_dims(test, axis=0)
    p = model.predict(test)
    pmax = np.argmax(p)
    print ('Predicted probabilities: ', p*100)
    print('Predicted class: ', pmax)
    if pmax == y_test[rand]:
        a = 'Correct prediction'
        print(a)
    else:
        a = 'Incorrect prediction'
        print(a)
    plt.subplot(10,2,j+1)  # Plot the image
    plt.imshow(X_test[rand], cmap='gray', )
    plt.title('\n Image to be Predicted with label: {} '.format(y_test[rand]))
    plt.axis('off')
    plt.subplot(10,2,k+1)  # Plot the predicted class
    i = np.max(np.where(y_test==pmax))
    plt.imshow(X_test[i], cmap='gray')
    plt.title('\n Predicted class: {}   '.format(pmax) + '-----> {}'.format(a))
    plt.axis('off')
    plt.subplots_adjust(hspace= 0.390)
    j= j+2
    k= k+2
plt.show()

# The model perform very well with accuracy well over 90% and most of the time give 100 face match. 

#################### END of Part 1 ##########################################


# PART 2 : Training with Transfer Learning
# Applying transfer learning from Googles ResNet CNN model
# We will use Keras implementation of ResNet 50V2 with weights trained on imagenet as a pretrained base model.
# This model will be used as a base model with pretrained weights that are frozen (Not Trainable weights)
# However base model will be extended with additional trainable dense layers.
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(112,92,3),
)

# Freeze the base_model
base_model.trainable = False

# Create new model on top
i = tf.keras.Input(shape=(112, 92, 3))

# Rescaling by adding a scaling layer
scale_layer = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
Scaled_i = scale_layer(i)

# Adding the scaled layer to base model
o = base_model(Scaled_i, training=False)
o = tf.keras.layers.GlobalAveragePooling2D()(o)
o = tf.keras.layers.Dropout(0.2)(o)
# Adding trainable layers to the base model
o = tf.keras.layers.Flatten()(o)
o = tf.keras.layers.Dense(512, activation='relu')(o)
o = tf.keras.layers.Dropout(0.4)(o)
o = tf.keras.layers.Dense(20, activation='softmax')(o)

# Add the input shape to the model
TL_model = tf.keras.Model(inputs=i, outputs=o)

# Compile the model
TL_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
TL_model.save_weights('TLbest_model.h5')
TL_model.summary(show_trainable=True)

# Reshaping and Rescaling the datasets needed for Resnet compatiblility ie. having 4 dimensions and the channel dimension having 3 channel shape for RGB from 1 channel grayscale
TL_trainX =  np.reshape(image_train,newshape=(data['trainX'].shape[0],112,92, 1))
TL_trainX = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB), np.float32(TL_trainX))))/255.
print('Shape of TL_trainX: ', TL_trainX.shape)
i=0
j=0
plt.figure(figsize=(10,10))
for i in range(20):
    i = np.max(np.where(label_train==j)) # find location index with value corresponding to values from 0 to 19
    plt.subplot(4,5,j+1)
    plt.imshow(TL_trainX[i])
    plt.title(label_train[i])
    plt.axis('off')
    j= j+1
plt.show()
TL_testX =  np.reshape(image_test,newshape=(data['testX'].shape[0],112,92, 1))
TL_testX = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB), np.float32(TL_testX))))/255.


# Spliting the testing dataset for 
TLX_test, TLX_val, TLy_test, TLy_val = train_test_split(TL_testX, label_test, test_size=0.5, random_state=42, stratify=label_test)

# Assinging new variables for training dataset
TLX_train = TL_trainX
TLy_train = label_train

# Number of training vs validation dataset of training dataset and testing dataset
print('Training data size: ', TLX_train.shape[0])
print('Validation data size: ', TLX_val.shape[0])
print('Training data label size: ', TLy_train.shape[0])
print('Validation data label size: ', TLy_val.shape[0])
print('Testing data size: ', TLX_test.shape[0])
print('Testing data label size: ', TLy_test.shape[0])


# Training the model using same parmaeters as trained model without TL
TL_history = TL_model.fit(
    x=TLX_train,
    y=TLy_train,
    validation_data= (TLX_val, TLy_val),
    epochs=200,
    batch_size=32,
    callbacks=[   # callback to save best accuracy model weights and reduce learning rate if loss remains same
        tf.keras.callbacks.ModelCheckpoint(filepath="TLbest_model.h5",monitor="val_accuracy",verbose=1,save_best_only=True,save_weights_only=True,
        mode="auto",save_freq="epoch",initial_value_threshold=None),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=20,min_lr=0.0001,factor=0.1)])

# Save the model
TL_model.save('TLmy_model')

# Load the model weights with best validation accuracy
TL_model.load_weights('TLbest_model.h5')

# Evaluate the model on the test data
results = TL_model.evaluate(x=TLX_test, y=TLy_test, verbose = 0)
print('Test for TL model Loss: {:.5f}'.format(results[0]))
print('Test for TL model Accuracy: {:.2f}%'.format(results[1] * 100))

# The model performs good giving high accuracy. However, lacks behind the Non-Transfer Learning model from part 1, even for double the training epochs. 
# The issue is most likely with the competiblity of the ResNet model used which is trained on coloured images. And since we just reshaped the images to RGB 3 channels, this
# could cause the pre-trained model to learn the features of the grayscaled images.