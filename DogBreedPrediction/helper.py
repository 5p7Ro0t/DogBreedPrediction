import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_hub as hub

unique_breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
                 'american_staffordshire_terrier', 'appenzeller',
                 'australian_terrier', 'basenji', 'basset', 'beagle',
                 'bedlington_terrier', 'bernese_mountain_dog',
                 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
                 'bluetick', 'border_collie', 'border_terrier', 'borzoi',
                 'boston_bull', 'bouvier_des_flandres', 'boxer',
                 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
                 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
                 'chow', 'clumber', 'cocker_spaniel', 'collie',
                 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
                 'doberman', 'english_foxhound', 'english_setter',
                 'english_springer', 'entlebucher', 'eskimo_dog',
                 'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
                 'german_short-haired_pointer', 'giant_schnauzer',
                 'golden_retriever', 'gordon_setter', 'great_dane',
                 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
                 'ibizan_hound', 'irish_setter', 'irish_terrier',
                 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
                 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
                 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
                 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
                 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
                 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
                 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
                 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
                 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
                 'saint_bernard', 'saluki', 'samoyed', 'schipperke',
                 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
                 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
                 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
                 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
                 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
                 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
                 'west_highland_white_terrier', 'whippet',
                 'wire-haired_fox_terrier', 'yorkshire_terrier']


def get_pred_label(prediction_probabilities):
    """ 
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]



# Define image size
IMG_SIZE = 224
def process_image(image_path):
    """
    Takes an image file path and turns the image into a Tensor.
    """

    # Read in an image file
    image = tf.io.read_file(image_path)

    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green,Blue)
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert the colour channel values from 0-255 to 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to our desired value(224, 224)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image



def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image,label)
    """
    image = process_image(image_path)
    return image, label


# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn data into batches


def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as input (no labels).
    """
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    elif valid_data:
        print("Creating validation data batches")
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        print("Creating training data batches...")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X), tf.constant(y)))

        # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(X))

        # Create ( image, label ) tuples (this also turns the image path into a preprocessed image)
        data = data.map(get_image_label)

        # Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)
        return data_batch


# Create a function to load a trained model
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from : {model_path}")
  model = tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})
  return model