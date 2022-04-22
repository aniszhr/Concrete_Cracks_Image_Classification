# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:53:00 2022

@author: ANEH
"""

import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt

#1. Load image files
file_path = r"C:\Users\ANEH\Documents\Deep Learning Class\TensorFlow Deep Learning\Datasets\concrete_crack"
data_dir =  pathlib.Path(file_path)
BATCH_SIZE = 16
SEED = 12345
IMG_SIZE = (160,160)

#%%
#2. Split intp train-validation set
train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                            validation_split=0.2,
                                                            subset='training',
                                                            shuffle=True,
                                                            seed=SEED,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                          validation_split=0.2,
                                                          subset='validation',
                                                          shuffle=True,
                                                          seed=SEED,
                                                          batch_size=BATCH_SIZE,
                                                          image_size=IMG_SIZE)

#%%
#3.To obtain validation and test data, split validation set
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)


#%%
#Create prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset_pf = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)

#Data prep is done


#%%
#4.
data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip('horizontal'))
data_augmentation.add(tf.keras.layers.RandomRotation(0.2))


#%%
#4.1 Define preprocess inputs for transfer learning model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#Create the base model by calling out MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

#%%
#4.2 Create our own classification layer
class_names = train_dataset.class_names
global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
output_dense = tf.keras.layers.Dense(len(class_names),activation='softmax')


#%%
#Use functional API to construct the entire model (augmentation + preprocess input + CNN)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_avg_pool(x)
outputs = output_dense(x)

model = tf.keras.Model(inputs,outputs)
#Print model structure
model.summary()

#%%

#Compile model
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=adam,loss=loss,metrics=['accuracy'])

#%%
#5. Perform training
EPOCHS = 10
import datetime
log_path = r'C:/Users/ANEH/Documents/Deep Learning Class/TensorFlow Deep Learning/TensorBoard/project3_log'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
history = model.fit(train_dataset_pf,validation_data=validation_dataset_pf,epochs=EPOCHS,callbacks=[tb_callback])

#%%
#5. Evaluate with test dataset
test_loss,test_accuracy = model.evaluate(test_dataset_pf)

print('------------------------Test Result----------------------------')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
#6. Deploy model to make prediction
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions,axis=1)

#%%
#7. Show some prediction results
import os

plt.figure(figsize=(10,10))

for i in range(4):
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')
    
save_path = r"C:\Users\ANEH\Documents\Deep Learning Class\AI05\AI05_repo_3\img"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()