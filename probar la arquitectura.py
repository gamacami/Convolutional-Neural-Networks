from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as keras_image
import keras.utils as image
import matplotlib.pyplot as plt
import numpy as np

loaded_model = load_model('women_or_men_model_3.h5')

img_path = 'Trabajo final/imagenesdeprueba/women_or_men_8.jpeg'
img = image.load_img(img_path, target_size=(64, 64))

img_array = image.img_to_array(img)

plt.figure(1)
plt.imshow(img_array/255)
plt.show()

ImageT = np.expand_dims(img_array, axis = 0)
Women_or_men = loaded_model.predict(ImageT)
print(ImageT)
threshold = 0.5
if Women_or_men[0][0] > threshold:
    print('Man')
elif Women_or_men[0][0] < (1 - threshold):
    print('Woman')
else:
    print('Queer')
