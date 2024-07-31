import keras 
import keras
import numpy as np
import random
import os,cv2
from utils.image_utils import load_images
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf


SEED_VALUE=42 
random.seed(SEED_VALUE)

def CNN():
    img_size = 300
    model = keras.Sequential()
    # First Convolutional Block
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    # Second Convolutional Block
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    # Third Convolutional Block
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    # Add Dense and flatten layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(4, activation='softmax'))
    return model


def model_train(model_dir,history_dir,epochs=20):
    batch_size=16
    img_size=300
    if not os.path.exists(model_dir):
        path_train='C:/Users/brian/Desktop/DLAssignment/brain-tumor-classification-dataset/Training'
        path_test='C:/Users/brian/Desktop/DLAssignment/brain-tumor-classification-dataset/Testing'
        #Load training and testing data
        train_img,train_labels = load_images(path_train, img_size)
        test_img,test_labels=load_images(path_test, img_size)
        train_img=np.array(train_img)
        test_img=np.array(test_img)
        train_img, train_labels = sklearn.utils.shuffle(train_img, train_labels, random_state=42)
        X_train, X_test, Y_train, Y_test = train_test_split(train_img,train_labels,test_size = 0.1)
        #compile the model
        model=CNN()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #set up the checkpoint 
        checkpoint = keras.callbacks.ModelCheckpoint(f"{model_dir}".split('.')[0]+".keras",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
        #Set up early stopping 
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='auto', restore_best_weights=True)
        #change learning rate if validation accuracy doesnt change by much
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,mode='auto',verbose=1)      
        #data augmentation to training data
        img_datagen =tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,horizontal_flip=True)
        img_datagen.fit(X_train)
        #Train the model
        history = model.fit(img_datagen.flow(tf.cast(X_train, tf.float32), np.array(pd.get_dummies(Y_train)),shuffle=True), validation_data=(tf.cast(X_test, tf.float32), np.array(pd.get_dummies(Y_test))), epochs =epochs, verbose=1, batch_size=batch_size,  callbacks=[early_stopping, checkpoint,reduce_lr])
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(history_dir, index=False)
        return model, history_df
    else:
        # load the history
        history_df = pd.read_csv(history_dir)
        model_loaded= tf.keras.models.load_model(model_dir)
        # evaluate the model
        model_loaded.evaluate(tf.cast(X_test, tf.float32), np.array(pd.get_dummies(Y_test)))
        return model_loaded, history_df

model_train('training_result','metrics_history',epochs=20)

  

    

       



