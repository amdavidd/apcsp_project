import pandas as pd

# Extract feature names from the .names file
features = []
with open('Dataset/communities.names', 'r') as f:
    for line in f:
        if line.startswith('@attribute'):
            # Get the attribute name (second word)
            features.append(line.split()[1])

# Add the column names to the DataFrame
input_file = 'Dataset/communities.data'
output_file = 'Dataset/crimedata.csv'

df = pd.read_csv(input_file, names=features, na_values='?')

df.to_csv(output_file, index=False)
print('Conversion complete!')