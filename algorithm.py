import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

genData = pd.read_csv("Dataset/name_gender_dataset.csv")
genData.drop(['Probability'], axis = 1, inplace = True)
print(genData.head())

model = LogisticRegression()

train_data = genData['name']
train_labels = genData['gender']


model.fit(train_data, train_labels)
