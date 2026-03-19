import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# dataset path
dataset_path = (r"C:\Users\jatin\OneDrive\Documents\GitHub\SCT_ML_03\cats_vs_dogs_svm\dataset")

categories = ["cats", "dogs"]

data = []
labels = []

IMG_SIZE = 64

# Load images
for category in categories:
    folder = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)

        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            data.append(image.flatten())
            labels.append(label)

        except:
            pass

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Dataset Loaded:", data.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel='linear')

print("Training model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))