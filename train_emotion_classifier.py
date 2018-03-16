from keras.layers import Dense, Flatten
from keras.models import Sequential
import pandas as pd
import wandb
import numpy as np
import cv2
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config
# parameters
config.batch_size = 32
config.num_epochs = 10000
input_shape = (64, 64, 1)

wandb_callback=  WandbKerasCallback(save_model=False)


def load_fer2013():
    
    data = pd.read_csv("datasets/fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]
    
    return train_faces, train_emotions, val_faces, val_emotions

# loading dataset

train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape

model = Sequential()
model.add(Flatten(input_shape=(48,48,1)))
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])

model.fit(train_faces, train_emotions, batch_size=config.batch_size,
                    epochs=config.num_epochs, verbose=1, callbacks=[wandb_callback],
                    validation_data=(val_faces, val_emotions))





