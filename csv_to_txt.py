import pandas as pd
# convert .txt data file to .csv
input_file = ''
output_file = 'Dataset/year_predictions.csv'

# Read the text file directly into a DataFrame
df = pd.read_csv(input_file, delimiter=',')  # Adjust delimiter as needed

# Write the DataFrame to a CSV file
df.to_csv(output_file, index=False)

print('Conversion complete!')