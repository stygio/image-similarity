# Import necessary libraries
import numpy as np
from numpy.linalg import norm
import keras
from keras.applications.efficientnet import EfficientNetB3
from keras.applications.efficientnet import preprocess_input
import os
from PIL import Image
from sys import argv

if len(argv) < 2:
    raise Exception('Please pass path to directory of submitted photos as an argument.')

ORIGINALS_DIR = './originals'
SUBMISSIONS_DIR = argv[1]

# Cosine similary between 1d vectors
def cosine(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

# Function to extract features using EfficientNetB3 and calculate cosine similarity
def calculate_feature_similarity(image_path1, image_path2):
    # Function to preprocess image for EfficientNetB3
    def preprocess_image(image_path, usePadding = False):
        # Target size for EfficientNetB3 input is 300x300
        targetSize = 300
        if usePadding:
            img = keras.utils.load_img(image_path)
            x, y = img.size
            # Find square size
            size = max(targetSize, x, y)
            inputImg = Image.new('RGB', (size, size), (0, 0, 0))
            inputImg.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        else:
            inputImg = keras.utils.load_img(image_path, target_size = (targetSize, targetSize))
        # inputImg.show()
        # input()
        data = keras.utils.img_to_array(inputImg)
        data = np.expand_dims(data, axis=0)
        data = preprocess_input(data)
        return data

    # Load pre-trained EfficientNetB3 model
    model = EfficientNetB3(weights='imagenet', include_top=False)
    # Preprocess the two images
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)
    # Extract features
    features_img1 = model.predict(img1)
    features_img2 = model.predict(img2)
    # Flatten features and calculate cosine similarity
    features_img1 = features_img1.flatten()
    features_img2 = features_img2.flatten()
    similarity = cosine(features_img1, features_img2)
    similarity = f'{similarity * 100:.2f}'
    return similarity

# Create table for mapping
originals = sorted([file for file in os.listdir(ORIGINALS_DIR) if file != '.DS_Store'])
table = [[os.path.splitext(file)[0], os.path.join(ORIGINALS_DIR, file)] for file in originals]
submissions = [file for file in os.listdir(SUBMISSIONS_DIR) if file != '.DS_Store']
for submission in submissions:
    submissionLabel = os.path.splitext(submission)[0]
    for entry in table:
        if entry[0] == submissionLabel:
            entry.append(os.path.join(SUBMISSIONS_DIR, submission))
            continue

# Calculate feature-based similarity using EfficientNetB3 for each pair
for entry in table:
    entry.append(calculate_feature_similarity(entry[1], entry[2]))

# Pretty print table of results
for row in table:
    for col in row:
        print(f'{col}', end = ' ')
    print('')

# Calculate average score
avgScore = sum([float(entry[-1]) for entry in table])/len(table)
print(f'Average score for submissions: {avgScore:.2f}')
