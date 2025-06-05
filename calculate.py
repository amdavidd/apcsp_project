from algorithm import forest, linear, neighbors

import json

# Read the JSON file
with open('user_input.json', 'r') as file:
    input_array = json.load(file)

# Print the array
print(input_array)

# Example input array for predictions
input_array = [input_array]  # Replace with your input data, e.g., [value1, value2, ..., valueN]

# Make predictions using all three models
forest_prediction = forest.predict(input_array)
linear_prediction = linear.predict(input_array)
knn_prediction = neighbors.predict(input_array)

print('Random Forest Prediction:', forest_prediction)
print('Linear Regression Prediction:', linear_prediction)
print('KNN Prediction:', knn_prediction)

predictions = [
    float(forest_prediction[0]),  # Extract the first value from the list
    float(linear_prediction[0]),   # Extract the first value from the list
    float(knn_prediction[0])       # Extract the first value from the list
]

# Write to JSON file
with open('predictions.json', 'w') as json_file:
    json.dump(predictions, json_file, indent = 4)

print("Predictions saved to predictions.json")