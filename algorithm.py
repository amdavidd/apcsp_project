import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


#get csv data
df = pd.read_csv('Dataset/year_predictions_csv')
print(df.head())

# #prepare data
# train_data = genData['Name']
# train_labels = genData['Gender']


# #use label encoder to convert it into numerical values
# label_encoder = LabelEncoder()
# train_data_encoded = label_encoder.fit_transform(train_data)
# train_data_encoded = train_data_encoded.reshape(-1, 1)

# feature_train, feature_test, label_train, label_test = train_test_split(train_data_encoded, train_labels, test_size = 0.25, random_state = 42)


# #fit the model
# model = LogisticRegression()
# model.fit(feature_train, label_train)

# label_predicted = model.predict(feature_test)
# print('predicted classes: ', label_predicted)

# print('True classes:', label_test)

# print(confusion_matrix(label_predicted, label_test))




# print('done')




