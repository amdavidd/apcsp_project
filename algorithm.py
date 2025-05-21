import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



#get csv data
features = pd.read_csv('Dataset/cleaned_crimedata_features.csv')
target = pd.read_csv('Dataset/cleaned_crimedata_target.csv')

#print(features.dtypes)
#print(target)


feature_train, feature_test, label_train, label_test = train_test_split(features, target, test_size = 0.2, random_state = 42)

forest = RandomForestRegressor(n_estimators=1000, random_state=42)

forest.fit(feature_train, label_train.values.ravel())

#print feature importances
importances = pd.Series(forest.feature_importances_, index=features.columns)
print(importances.sort_values(ascending=False))

#get features
label_predicted = forest.predict(feature_test)
#print('predicted classes: ', label_predicted)
#print('True classes:', label_test.values.ravel())

print(forest.score(feature_test, label_test.values.ravel()))

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#fit the model
model = LinearRegression()
model.fit(feature_train, label_train.values.ravel())

label_predicted = model.predict(feature_test)
print('Predicted values:', label_predicted)
print('True values:', label_test.values.ravel())

print('Mean Squared Error:', mean_squared_error(label_test.values.ravel(), label_predicted))
print('Mean Absolute Error:', mean_absolute_error(label_test.values.ravel(), label_predicted))
print('R^2 Score:', r2_score(label_test.values.ravel(), label_predicted))

# Display feature importance (absolute value of coefficients)
feature_importance = pd.Series(abs(model.coef_), index=features.columns)
print("Feature importance (by absolute coefficient value):")
print(feature_importance.sort_values(ascending=False))