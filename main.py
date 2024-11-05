# Importing Libraries:
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost.testing.data import joblib

# for displaying all feature from dataset:
pd.pandas.set_option('display.max_columns', None)

# Reading Dataset:
dataset = pd.read_csv("Liver_data.csv")

# Filling NaN Values of "Albumin_and_Globulin_Ratio" feature with Median:
dataset['Albumin_and_Globulin_Ratio'] = dataset['Albumin_and_Globulin_Ratio'].fillna(
    dataset['Albumin_and_Globulin_Ratio'].median())

# Label Encoding:
dataset['sex'] = np.where(dataset['sex'] == 'Male', 1, 0)
dataset['Ascites'] = np.where(dataset['Ascites'] == 'Y', 1, 0)
dataset['Hepatomegaly'] = np.where(dataset['Hepatomegaly'] == 'Y', 1, 0)
dataset['Spiders'] = np.where(dataset['Spiders'] == 'Y', 1, 0)
dataset['Edema'] = np.where(dataset['Edema'] == 'Y', 1, 0)

# Droping 'Direct_Bilirubin' feature:
# dataset = dataset.drop('Direct_Bilirubin', axis=1)

# Independent and Dependent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# SMOTE Technique:w
from imblearn.combine import SMOTETomek

smote = SMOTETomek()
X_smote, y_smote = smote.fit_resample(X, y)

# Train Test Split:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=33)

# ---------------------------------------------RandomForestClassifier--------------------------------------------:
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train, y_train)

# rf_preds = RandomForest.predict(X_test)

# Creating a pickle file for the classifier
# filename = 'Liver2.pkl'
# pickle.dump(RandomForest, open(filename, 'wb'))

# ---------------------------------------------logistic regression-----------------------------------------------:
lr = LogisticRegression()
lr.fit(X_train, y_train)

# lr_preds = lr.predict(X_test)

# filename1 = 'LiverLogistic.pkl'
# pickle.dump(lr, open(filename1, 'wb'))

# ----------------------------------------- GradientBoosting Classifier----------------------------------------- :
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)

# gb_preds = gb_classifier.predict(X_test)

# filename2 = 'gradientBoosting.pkl'
# pickle.dump(gb_classifier, open(filename2, 'wb'))


classifiers = {
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

accuracies = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    print(f'{name} Accuracy: {accuracy}')

# Select the best model based on accuracy
best_model_name = max(accuracies, key=accuracies.get)
best_model = classifiers[best_model_name]

# Save the best model to a pickle file
joblib.dump(best_model, 'best_model.pkl')






# greatest = np.array(
#     [accuracy_score(y_test, rf_preds), accuracy_score(y_test, lr_preds), accuracy_score(y_test, gb_preds)])
# #
# # print(rf_preds, lr_preds, gb_preds)
# # value = max(greatest)
#
#
# print(f"Accuracy: {accuracy_score(y_test, rf_preds)}")
#
# print("logistic regression:")
# print(classification_report(y_test, lr_preds))
# print(f"Accuracy: {accuracy_score(y_test, lr_preds)}")
#
# print("GradientBoosting Classifier:")
# print(classification_report(y_test, gb_preds))
# print(f"Accuracy: {accuracy_score(y_test, gb_preds)}")
#
# # greatest = rf_preds if rf_preds >= lr_preds and rf_preds >= gb_preds else (lr_preds if lr_preds >= gb_preds else gb_preds)
# if greatest == rf_preds:
#     sample = filename
# elif greatest == lr_preds:
#     sample = filename1
# else:
#     sample = filename2
