import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Function to extract features from an image
def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = gray_image.flatten()
    return features

# Assuming steganographic and non-steganographic images are in respective folders
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

stego_images = load_images_from_folder('stego_images/')
non_stego_images = load_images_from_folder('non_stego_images/')

# Extract features and labels
X = []
y = []

for img in stego_images:
    X.append(extract_features(img))
    y.append(1)  # Label 1 for steganographic images

for img in non_stego_images:
    X.append(extract_features(img))
    y.append(0)  # Label 0 for non-steganographic images

X = np.array(X)
y = np.array(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model for later use
with open('stego_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
