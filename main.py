from preprocessing import trs, tss
from model import neural_network
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np

training_set = trs()
test_set = tss()
cnn = neural_network()

cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=100
)
#The results I got after testing 3 times with 100 epochs was an avarege of 0.9603 and a maximum of 0.9626
#More tecnically 0.9603 +- 0.030

#Taking some especific images to evaluete the model
results = ["glioma", "meningioma", "no tumor", "pituitary"]
test_images = ["Data/Testing/glioma/Te-gl_0054.jpg", "Data/Testing/notumor/Te-no_0396.jpg", "Data/Testing/meningioma/Te-me_0020.jpg", "Data/Testing/pituitary/Te-pi_0016.jpg"]
for image in test_images:
    test_image = tf.keras.preprocessing.image.load_img(image, target_size=(64,64))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.0
    result = cnn.predict(test_image)
    result_index = np.argmax(result, axis=1)[0]
    result_name = results[result_index]
    print(result_name)