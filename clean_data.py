import pandas as pd

# Read the CSV file
df = pd.read_csv('Dataset/crimedata.csv')
print(df.head())

# Check for NaN values in the DataFrame
#cleaned_df = df.dropna(axis=1, how='any')
#print(cleaned_df.columns.to_list())
#print(cleaned_df.head())
#print(len(df.columns))
#print(df.isna().sum().to_string())

cleaned_df_features = df[['population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct16t24', 'agePct65up', 'pctUrban', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'perCapInc', 'PctPopUnderPov', 'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'TotalPctDiv', 'PersPerFam', 'PctRecImmig10', 'PctNotSpeakEnglWell', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'LandArea', 'PopDens']]   
cleaned_df_target = df['ViolentCrimesPerPop']
print(cleaned_df_features.isna().sum().to_string())
print(cleaned_df_features.head())

print(df.head())

# Save the cleaned DataFrame to a new CSV file
cleaned_df_features.to_csv('Dataset/cleaned_crimedata_features.csv', index=False)
cleaned_df_target.to_csv('Dataset/cleaned_crimedata_target.csv', index=False)