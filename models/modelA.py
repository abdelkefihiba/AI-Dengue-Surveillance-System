#MODEL A - Mosquito Species Classifier (HOG + SVM)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#LOAD PREPROCESSED DATASET
print("Loading dataset")
#load HOG Features (X) and labels (y)
X = pd.read_csv("hog_features.csv").values
y = pd.read_csv("labels.csv").values.ravel()

print("Dataset loaded successfully!")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

#SPLIT INTO TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#CREATE SVM MODEL
print("\nCreating SVM model...")

model = SVC(kernel="linear", probability=True)

print("Model created!")

#TRAIN MODEL
print("\nTraining the model...")

model.fit(X_train, y_train)

print("Training complete!")

#EVALUATE THE MODEL
print("\nEvaluating model...")

y_pred = model.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

#SAVE THE TRAINED MODEL
print("\nSaving the trained model...")

joblib.dump(model, "mosquito_svm_model.pkl")

print("Model saved as mosquito_svm_model.pkl \nModel A training completed successfully!")