import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, 
    roc_curve, auc, precision_score, recall_score)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

data = pd.read_csv(r'C:\Users\Lenovo\Desktop\machineLearningFiles\data\healthCareData\heart.csv')

# Heart Disease Classification
# Age | Sex | Chest Pain Type with 4 values | resting blood pressure |serum cholestoral in mg/dl
# fasting blood sugar > 120 mg/dl | resting electrocardiographic results with values of 0,1,2
# maximum heart rate achieved | exercise induced angina | oldpeak = ST depression induced by exercise relative to rest
# the slope of the peak exercise ST segment
# number of major vessels 0-3 colored by floursopy
# thal: 3 = normal; 6= fixed defect; 7=reversable defect

x = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

ss = StandardScaler()
x = ss.fit_transform(x)

xTrain, xTest, yTrain, yTest = tts(x, y, test_size=0.2, random_state=109)

model = SVC(kernel='linear')
model.fit(xTrain, yTrain)

yPred = model.predict(xTest)
print('Classifier Performance = ', accuracy_score(yTest, yPred))
print('\nConfusion Matrix\n', confusion_matrix(yTest, yPred))
print(precision_score(yTest, yPred))
print(recall_score(yTest, yPred))
fpr, tpr, thresholds = roc_curve(yTest, yPred)
print('\n\nAUC: ', auc(fpr, tpr))