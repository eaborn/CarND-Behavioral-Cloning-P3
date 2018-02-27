import csv
import cv2
import numpy as np
from sklearn.utils import shuffle


lines = []
with open('/home/carnd/udacity/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # ignore the header
    next(reader,None)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = '/home/carnd/udacity/data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
            #consider steering angle as measurement
                correction = 0.28
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(angle+correction)
                angles.append(angle-correction)

            augmented_images, augmented_angles = [], []
            for image_new, angle_new in zip(images, angles):
                augmented_images.append(image_new)
                augmented_angles.append(angle_new)
                augmented_images.append(cv2.flip(image_new,1))
                augmented_angles.append(angle_new * (-1))

            yield shuffle(np.array(augmented_images), np.array(augmented_angles))


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras import backend as K



model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model.compile(loss='mse', optimizer='adam')

#from keras.models imort Model
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)


model.save('model.h5')
print(history_object.history.keys())


plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('model.png')
K.clear_session()

