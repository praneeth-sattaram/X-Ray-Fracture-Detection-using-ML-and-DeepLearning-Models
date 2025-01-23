!pip install tensorflow

import os, torch, shutil 
import numpy as np from glob 
import glob from PIL 
import Image
from torch.utils.data 
import random_split, Dataset, DataLoader 
from torchvision 
import transforms as T 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns 
import pandas as pd 
import os
from glob import glob

from sklearn.metrics 
import confusion_matrix, classification_report 
from tqdm.notebook import tqdm 
from sklearn.metrics import f1_score
from torchvision. transforms.functional import normalize, resize, to_pil_image 
import cv2 as cv 
from scipy.spatial.distance import cdist 
from keras import utils 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.keras.layers import * 
import tensorflow as tf

data_path = "/kaggle/input/bone-break-classification-image-dataset/Bone Break Classification/BoneBreak Classification"
image_paths = glob(data_path+///*)
data_df = pd. DataFrame( ('image_path': image_paths))
data_fl'id'] = data_df.image_path.apply(lambda x: x.split('/')[-1][:-4])
data_df[ 'set'] = data_df.image_path.apply(lambda x: x. split (*/')[-2])
data_df['target_name'] = data_df.image_path.apply(lambda x: x.split(*/')[-3])
catg_df = pd. Categorical(data_df[' target_name' ])
target_names, data_df[ 'target'] = catg_df.categories, catg_df.codes
data_df.head()

fig, axes = plt. subplots(10,5, figsize=(15,30))
def kriging interpolation (image, window_size=3):
    def estimate_pixel_value(neighbors, distances):
         weights = 1 / (distances + 1e-10) # Add small value to avoid division by zero
         weights /= np. sum(weights)
         estimated_value = np. sum(weights * neighbors)
         return estimated_value
    def is noisy_pixel(pixel_value):
         return pixel_value == 0 or pixel_value == 255
         
    padded_image = np. pad (image, pad_width=window_size//2, mode=' edge' )
    result_image = np.copy(image)

for i in range (image. shape [0]) : 
  for j in range (image. shape [1]) :
     if is_ noisy_pixel(image[i, j]):
        window =padded_image[i:i+window_size, j:j+window_size].flatten()
        non_noisy indices = np.where ((window != 0) & (window != 255))[0]
        
        if len(non_noisy_indices) > 0:
           neighbors = window[non_noisy_indices]
           distances = cdist([(window_size//2, window_size//2)],[(idx // window_size, id % window_size) for idx in non_noisy-indices,metric= euclidean) .flatten()
           result image[i, j] = estimate_pixel_ value (neighbors, distances)
        else:
           result_image[i, j] =image[i, j]

    return result_image 
for ax_row, target_name in zip(axes, target_names):
    sample_df = data_df[data_df. target_name == target_name]. sample(n=5)
    for nidx, ax in enumerate(ax_row) :
        # _image = Image.open(sample_df.image_path.iloc[nidx])
        img = cv. imread(sample_df.image_path.iloc[nidx])
        assert img is not None, "file could not be read, check with os.path.exists()"
        kernel = np. ones ((5, 5), np. float32) /25
        dst = cv. filter2D(img, -1, kernel)
        edges = cv. Canny (dst, 50,25)
        ax. imshow(edges, cmap = "gray")
        ax. axis('off')
        ax. set_title(target_name)
        
plt. tight_layout ()
plt. savefig('demo_fig.png')

train_data-utils.image_dataset_from_directory(
data_path, labels="inferred" label_mode= "int", validation_split=0.1, subset="training", shuffle-True, color_mode="rgb",
image_size= (256,256), batch_size=64, seed=40,)

vald_data=utils.image_dataset_from_directory(
data_path, labels="inferred" label_mode="int" validation_split=0.1, subset="validation"
color_mode="rgb"
image_size=(256,256), batch_size=64,
seed=40,)

#normalization
classes = {
'Avlusion fracture': 0,
'Comminuted fracture': 1,
'Fracture Dislocation': 2,
'Greenstick fracture': 3,
'Hairline Fracture': 4,
' Impacted fracture': 5,
'Longitudinal fracture': 6,
'Oblique fracture': 7,
'Pathological fracture': 8,
'Spiral Fracture': 9
}

def.normalize (image, y):
    return image/255.0, y
train_data = train_data.map(normalize)
vald_data = vald_data.map(normalize)
train_x=[]
train_y=[]
for image label in train_data:
    train_append (image)
    train_y append (label)
    print (type(train_y))
train_x = tf .concat(train_x, axis=0)
train_y = tf.concat (train_y, axis=0)
val_x= []
val_y=[]
for image, label in train_data:
    val_x. append (image)
    val-y. append (label)
val_x = tf. concat(val_x, axis=0)
val_y = tf .concat(val_y, axis=0)
num_classes = 10
train_y = tf.keras.utils.to_categorical(train_y. num_classes=num_classes)
val_y = tf.keras.utils.to_categorical(val_y, num_classes=num_classes)

train_datagen = tf.keras.preprocessing. image.ImageDataGenerator(
rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
val_datagen = tf.keras.preprocessing. image.ImageDataGenerator()
Mrain_ generator = train_datagen.flow(x=train_x, y=train_y, batch_size=64)
val_generator = val_datagen. flow(x=val_x, y=val_y, batch_size=64)
class_labels=["Avulsion fracture", "Comminuted fracture", "Fracture Dislocation", "Greenstick fracture","Hairline Fracture","Impacted fracture","Longitudinal fracture"
,"Oblique fracture","athological fracture", "Spiral Fracture"]
# Initialize the figure and subplots
fig,axes = plt.subplots(2, 4, figsize=(15, 5))
# Iterate through the first 10 images
for i, ax in enumerate(axes.flat):
# Select the image and label
image, label = train_[il, train_yli]
ax. imshow(edges, cmap='gray')
# Set the title with the class label
ax. set_title(f"{class_labels[np.argmax(label) ]}")
ax. axis( 'off')
# Display the figure
plt. show()
input_shape = (256, 256, 3)
num_classes = 10 # Replace with the actual number of classes
# Model architecture
def create_model(input_shape, num_classes):
    model = Sequential()
# First convolutional block
    model add (Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model. add (BatchNormalization))
    model. add (MaxPooling2D((2, 2)))
    model. add (Dropout (0.25))

# Secona convolutional pLOCK
    model add (Conv2D(64, (3, 3), activation='relu'))
    model. add (BatchNormalization)) 
    model. add (MaxPooling2D((2, 2)))
    model. add (Dropout (0.25))
# Third convolutional block
    model. add (Conv2D(128, (3, 3), activation=' relu'))
    model. add (BatchNormalization()) 
    model. add (MaxPooling2D((2, 2)))
    model. add (Dropout (0.25))
# Fourth convolutional block
    mode1. add (Conv2D(256, (3, 3), activation='relu'))
    model. add (BatchNormalization()) 
    model.add (MaxPooling2D((2, 2)))
    model. add{ Dropout(0.25))
# Fully connected layers
    model. add (Flatten())
    model. add (Dense(512, activation=' relu'))
    model. add (BatchNormalization())
    model. add (Dropout (0.5))
    model add (Dense (128, activation='relu'))
    model. add (BatchNormalization())
    model. add (Dropout (0.5))
# Output layer
    model. add (Dense (num_classes, activation=' softmax'))
    return model
# Create the model
model = create_model (input_shape, num_classes)
def visualize_augmented_data(dataset, num_images) :
    plt. figure(figsize=(10, 10))
    for images, labels in dataset. take(1) :
        for i in range (num_images) :
            ax = plt. subplot(4, 4, i + 1)
            plt. imshow(tf. squeeze(images[i]).numpy(), cmap='gray')
            plt. title(int(labels[i]))
            plt. axis("off")
# Visualize 16 images from the augmented training data
visualize_augmented_data(train_data, 16)
plt. show()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss= categorical_crossentropy',
              metrics=l'accuracy' ])

# Train the model
from tensorflow.keras.callbacks import EarlyStopping.
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(train_x, train_y, batch_size=10, epochs=1000,
          validation_data=(val_x, val_y), callbacks=[early_stopping])
# print(history history)
print ("Highest accuracy: ", max(history history[' accuracy' ]))
print("Highest validation accuracy: ", max(history history[' val_accuracy:']))
a = model.evaluate(val_x, val_y)
print (a)
#first value loss, second value accuracy
pred = model predict(val_x)
class_labels=["Avulsion fracture", "Comminuted fracture", "Fracture Dislocation", "Greenstick fracture","Hairline Fracture", "Impacted fracture", "Longitudinal fracture", "Oblique fracture",
"Pathological fracture", "Spiral Fracture")
num_images_to_display = 8
num_columns = 2
num_rows = (num_images_to_display + num_columns - 1)
fig, axes = plt/subplots(num_rows,num_columns, figsize=(15, 15))
for i< num_images_to_display:
    ax. imshow(val_x|il)
    actual_label = class_labels[np.argmax(val_y[il)]
    predicted_label = class_labels[np.argmax(pred[i])]
    ax. set_title(f"Actual: {actual_label), Predicted: (predicted_label)")

    ax.axis('off')
else:
    ax. axis('off')
plt. tight_layout ()
plt. show()
#save model
model. save("best_model")
def representative_data_gen():
    for images, - in train_data. take(100): # Use a small batch from your training dataset
        yield [tf. cast(images, tf.float32)]
# Convert the model to TensorFlow Lite format with int8 quantization
converter = tf.lite.TFLiteConverter. from_saved_model ("best_model")
converter optimizations = [tf.lite.Optimize.DEFAULT]
converter. representative_dataset = representative_data-gen
converter. target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter. inference_input_type = tf.uint8 # or tf.int8
converter.inference_output_type = tf.uint8 # or tf.int8
tflite_model = converter .convert)
# Save the quantized model to a file
with open("model_int8. tflite", "wb") as f:
     f. write(tflite_model)
import time
# Load the saved original TensorFlow model
original_model = tf.keras.models. load_model("best_model")
# Load the TFLite model
interpreter = tf. lite. Interpreter(model_path="model_int8.tflite") # Use the appropriate delegate
for your hardware
interpreter.allocate_tensors()
# Get input and output tensors for the FLite model
input_details = interpreter. get_input_details()
output_details = interpreter.get_output_details()
# Function to measure inference time for TensorFlow model
def measure_tf_inference_time(model, dataset) :
    start_time = time. time()
    predictions•= []
    for images, _ in dataset:
        predictions. append(model.predict(images))
    end_time = time. time ()
    return end_time - start_time, predictions
def measure_tflite_inference_time(interpreter, dataset) :
    input_details = interpreter get_input_details()
    output_details = interpreter.get_output_details)
    total_time = 0
    tflite_predictions = []
    for images, _ in dataset: 
        for image in images:
    # Add batch dimension
            image = np. expand_dims(image, axis=0) .astype(np.uint8)
            interpreter set_tensor (input_details[0][ 'index'], image)
            start_time = time. time()
            interpreter.invoke()
            end_time = time. time ()
            tflite_output = interpreter get_tensor(output_details[0][ 'index'])
            tflite_predictions.append (tflite_output)
        total_time+= end_time - start_time
    return total_time, tflite_predictions
original_inference_time, original_ predictions = measure_tf_inference_time(original_model, vald_data) #measure inference time?
print(f"Original TensorFlow model inference time: {original_ inference_time:.4f} seconds")
# Measure inference time for the FLite model
tflite_inference_time, tflite_predictions = measure_tflite_inference_time (interpreter, vald_data)
print(f"TFLite model inference time: {tflite_inference_time:.4f) seconds")
