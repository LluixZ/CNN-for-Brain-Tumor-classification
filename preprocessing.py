import tensorflow as tf
import os

main_dir = os.path.abspath(os.getcwd())
training_dir = os.path.join(main_dir, "Data/Training")
testing_dir = os.path.join(main_dir, "Data/Testing")

#The functions are just to use this code in another page, to make the project itself more readable
def trs():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        zoom_range = 0.2,
        horizontal_flip=True
    )
    training_set = train_datagen.flow_from_directory(
        training_dir,
        target_size=(64,64),
        batch_size=32,
        class_mode="categorical",
    )
    return training_set

def tss():
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255
    )
    test_set = test_datagen.flow_from_directory(
        testing_dir,
        target_size=(64,64),
        batch_size=32,
        class_mode="categorical",
    )
    return test_set