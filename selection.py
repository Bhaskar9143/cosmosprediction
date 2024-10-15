import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_excel('C:/Users/sreev/cosmospredict/datasetfordebris/fengyuan.xlsx')

# Convert 'EPOCH' to datetime and extract features
df['EPOCH'] = pd.to_datetime(df['EPOCH'], format='%Y-%m-%d %H:%M:%S.%f')
df['YEAR'] = df['EPOCH'].dt.year
df['MONTH'] = df['EPOCH'].dt.month
df['DAY'] = df['EPOCH'].dt.day
df['HOUR'] = df['EPOCH'].dt.hour
df['MINUTE'] = df['EPOCH'].dt.minute
df['SECOND'] = df['EPOCH'].dt.second

# Drop original 'EPOCH' column
df.drop(columns=['EPOCH'], inplace=True)

# Handle categorical variables
df = pd.get_dummies(df, columns=['CLASSIFICATION_TYPE'], drop_first=True)

# Define features and target variable
X = df.drop(columns=['OBJECT_NAME', 'OBJECT_ID', 'NORAD_CAT_ID'])  # Adjust as needed
y = df['MEAN_MOTION']  # Example target, adjust accordingly

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection using ANOVA
anova_selector = SelectKBest(score_func=f_regression, k='all')  # 'all' to see all features
anova_selector.fit(X_train_scaled, y_train)

# Get ANOVA scores and p-values
anova_scores = anova_selector.scores_
anova_pvalues = anova_selector.pvalues_

# Create a DataFrame for ANOVA results
anova_results = pd.DataFrame({
    'Feature': X.columns,
    'ANOVA Score': anova_scores,
    'p-value': anova_pvalues
})

# Filter significant features (e.g., p-value < 0.05)
significant_features = anova_results[anova_results['p-value'] < 0.05]

# Save selected features to a CSV file for later use in prediction
selected_feature_names = significant_features['Feature'].tolist()
with open('selected_features.txt', 'w') as f:
    for feature in selected_feature_names:
        f.write(f"{feature}\n")

print("ANOVA Results:\n", anova_results)
print("\nSignificant Features:\n", significant_features)

# RFE with Linear Regression on the selected features from ANOVA
X_train_selected = X_train[significant_features['Feature']]
X_test_selected = X_test[significant_features['Feature']]

# RFE using Linear Regression
model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)  # Adjust the number of features to select
rfe.fit(X_train_selected, y_train)

# Selected features from RFE
final_selected_features = X_train_selected.columns[rfe.support_]
print("Final Selected Features from RFE:", final_selected_features)

# Optionally save final selected features
with open('final_selected_features.txt', 'w') as f:
    for feature in final_selected_features:
        f.write(f"{feature}\n")
