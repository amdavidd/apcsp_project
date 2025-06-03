import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Get CSV data
features = pd.read_csv('Dataset/cleaned_crimedata_features.csv')
target = pd.read_csv('Dataset/cleaned_crimedata_target.csv')

# Split data into training and testing sets
feature_train, feature_test, label_train, label_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Random Forest Regressor
forest = RandomForestRegressor(n_estimators=1000, random_state=42)
forest.fit(feature_train, label_train.values.ravel())

# Predictions with Random Forest
label_predicted_forest = forest.predict(feature_test)
print('Random Forest R^2 Score:', forest.score(feature_test, label_test.values.ravel()))

# Linear Regression
linear = LinearRegression()
linear.fit(feature_train, label_train.values.ravel())

# Predictions with Linear Regression
label_predicted_linear = linear.predict(feature_test)
print('Linear Regression Mean Squared Error:', mean_squared_error(label_test.values.ravel(), label_predicted_linear))
print('Linear Regression Mean Absolute Error:', mean_absolute_error(label_test.values.ravel(), label_predicted_linear))
print('Linear Regression R^2 Score:', r2_score(label_test.values.ravel(), label_predicted_linear))

# KNN Regression
scores = []
for i in range(1, 100):
    classifier = KNeighborsRegressor(n_neighbors=i, weights="distance")
    classifier.fit(feature_train, label_train.values.ravel())
    scores.append(classifier.score(feature_test, label_test))

best_n_neighbors = scores.index(max(scores)) + 1  # +1 because index starts at 0
neighbors = KNeighborsRegressor(n_neighbors=best_n_neighbors, weights="distance")
neighbors.fit(feature_train, label_train.values.ravel())

# Predictions with KNN
label_predicted_knn = neighbors.predict(feature_test)
print('KNN Best n_neighbors:', best_n_neighbors)
print('KNN R^2 Score:', neighbors.score(feature_test, label_test.values.ravel()))

