import os
import cv2
import numpy as np
import warnings
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

print("Loading images...")

dataset_path = r"C:\Users\jatin\OneDrive\Documents\GitHub\SCT_ML_03\cats_vs_dogs_svm\dataset"

categories = ["cats", "dogs"]

data = []
labels = []

limit = 1000   # 1000 cats + 1000 dogs = 2000 images

for category in categories:

    folder = os.path.join(dataset_path, category)
    label = categories.index(category)

    count = 0

    for img in os.listdir(folder):

        if count >= limit:
            break

        img_path = os.path.join(folder, img)

        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (128,128))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                visualize=False
            )

            data.append(features)
            labels.append(label)

            count += 1

        except:
            pass


data = np.array(data)
labels = np.array(labels)

print("Dataset shape:", data.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("\nTraining SVM...\n")

model = SVC(kernel="rbf")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy,2))
print("\nClassification Report\n")

print(classification_report(y_test, predictions, target_names=["cats","dogs"], zero_division=0))