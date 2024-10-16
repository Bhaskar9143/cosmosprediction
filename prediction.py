import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
df = pd.read_excel('C:/Users/sreev/cosmospredict/datasetfordebris/fengyuan.xlsx')

# Load selected features
with open('final_selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f]

# Prepare data
X = df[selected_features]
y = df['MEAN_MOTION']

# Convert data types to save memory
for col in X.columns:
    if X[col].dtype == 'float64':
        X[col] = X[col].astype('float32')
    elif X[col].dtype == 'int64':
        X[col] = X[col].astype('int32')

y = y.astype('float32')

# Sample a fraction of the data for faster processing
sampled_df = df.sample(frac=0.1, random_state=42)
X = sampled_df[selected_features]
y = sampled_df['MEAN_MOTION']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM and GRU [samples, time steps, features]
X_train_scaled_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build and train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_scaled_reshaped.shape[1], X_train_scaled_reshaped.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))  # Output layer
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train LSTM using batch processing
batch_size = 32
for i in range(0, len(X_train_scaled_reshaped), batch_size):
    lstm_model.fit(X_train_scaled_reshaped[i:i + batch_size], y_train[i:i + batch_size],
                   validation_data=(X_test_scaled_reshaped, y_test), epochs=1, verbose=0, callbacks=[early_stopping])

# Predictions from LSTM
y_pred_lstm = lstm_model.predict(X_test_scaled_reshaped)

# Build and train GRU model
gru_model = Sequential()
gru_model.add(GRU(50, activation='relu', input_shape=(X_train_scaled_reshaped.shape[1], X_train_scaled_reshaped.shape[2])))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(1))  # Output layer
gru_model.compile(optimizer='adam', loss='mean_squared_error')

# Train GRU using batch processing
for i in range(0, len(X_train_scaled_reshaped), batch_size):
    gru_model.fit(X_train_scaled_reshaped[i:i + batch_size], y_train[i:i + batch_size],
                   validation_data=(X_test_scaled_reshaped, y_test), epochs=1, verbose=0, callbacks=[early_stopping])

# Predictions from GRU
y_pred_gru = gru_model.predict(X_test_scaled_reshaped)

# Combine predictions as new features for SVM
combined_predictions = np.column_stack((y_pred_lstm, y_pred_gru))

# Train SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(combined_predictions, y_test)

# Make predictions with SVM
y_pred_svm = svm_model.predict(combined_predictions)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred_svm)
print(f'Mean Squared Error of Hybrid Model (LSTM + GRU + SVM): {mse}')

# Function to calculate position using orbital elements
def calculate_position(df):
    df['INCLINATION'] = np.radians(df['INCLINATION'])
    df['RA_OF_ASC_NODE'] = np.radians(df['RA_OF_ASC_NODE'])
    df['ARG_OF_PERICENTER'] = np.radians(df['ARG_OF_PERICENTER'])
    df['MEAN_ANOMALY'] = np.radians(df['MEAN_ANOMALY'])

    # Calculate eccentric anomaly (E)
    def mean_to_eccentric_anomaly(M, e):
        E = M  # Initial guess
        for _ in range(10):
            E = M + e * np.sin(E)
        return E

    df['ECCENTRIC_ANOMALY'] = df.apply(lambda row: mean_to_eccentric_anomaly(row['MEAN_ANOMALY'], row['ECCENTRICITY']), axis=1)

    # Calculate true anomaly (ν)
    df['TRUE_ANOMALY'] = 2 * np.arctan(np.sqrt((1 + df['ECCENTRICITY']) / (1 - df['ECCENTRICITY'])) * np.tan(df['ECCENTRIC_ANOMALY'] / 2))

    # Calculate distance (r)
    df['ORBITAL_RADIUS'] = (1 - df['ECCENTRICITY'] ** 2) / (1 + df['ECCENTRICITY'] * np.cos(df['TRUE_ANOMALY']))

    # Calculate position in the orbital plane (x', y')
    df['x_prime'] = df['ORBITAL_RADIUS'] * np.cos(df['TRUE_ANOMALY'])
    df['y_prime'] = df['ORBITAL_RADIUS'] * np.sin(df['TRUE_ANOMALY'])

    # Calculate position in 3D space
    df['x'] = (df['x_prime'] * (np.cos(df['ARG_OF_PERICENTER']) * np.cos(df['RA_OF_ASC_NODE']) - 
                                 np.sin(df['ARG_OF_PERICENTER']) * np.sin(df['RA_OF_ASC_NODE']) * np.cos(df['INCLINATION'])))
    df['y'] = (df['x_prime'] * (np.cos(df['ARG_OF_PERICENTER']) * np.sin(df['RA_OF_ASC_NODE']) + 
                                 np.sin(df['ARG_OF_PERICENTER']) * np.cos(df['RA_OF_ASC_NODE']) * np.cos(df['INCLINATION'])))
    df['z'] = (df['x_prime'] * (np.sin(df['ARG_OF_PERICENTER']) * np.sin(df['INCLINATION'])))

    return df[['OBJECT_NAME', 'OBJECT_ID', 'x', 'y', 'z']]

# Calculate positions based on orbital parameters
predicted_positions_df = calculate_position(df)

# Function to check for potential collisions
def check_collision(predicted_positions_df, spacecraft_eq, spacecraft_size):
    collision_threshold = spacecraft_size + 10_000  # 10 km buffer in meters
    possible_collisions = []

    for index, row in predicted_positions_df.iterrows():
        # Calculate the spacecraft's position based on the given trajectory equation (example trajectory)
        t = index  # For simplicity, using index as time step
        spacecraft_x = spacecraft_eq['a1'] * t + spacecraft_eq['b1']
        spacecraft_y = spacecraft_eq['a2'] * t + spacecraft_eq['b2']
        spacecraft_z = spacecraft_eq['a3'] * t + spacecraft_eq['b3']

        # Calculate the distance between the debris and spacecraft
        distance = np.sqrt((row['x'] - spacecraft_x) * 2 + (row['y'] - spacecraft_y) * 2 + (row['z'] - spacecraft_z) ** 2)

        if distance <= collision_threshold:
            possible_collisions.append((row['OBJECT_NAME'], distance))

    return possible_collisions

# Sample spacecraft trajectory equation (example values)
spacecraft_trajectory_eq = {
    'a1': 500, 'b1': 1000,
    'a2': 600, 'b2': 1200,
    'a3': 700, 'b3': 1500
}
spacecraft_size = 15  # Example size of the spacecraft
possible_collisions = check_collision(predicted_positions_df, spacecraft_trajectory_eq, spacecraft_size)

# Print results
if possible_collisions:
    print("Potential collisions detected:")
    for collision in possible_collisions:
        print(f"Object {collision[0]} is within collision range, distance: {collision[1]:.2f} meters")
else:
    print("No potential collisions detected within the given range.")

# Save predicted positions to a CSV file
predicted_positions_df.to_csv('predicted_positions.csv', index=False)

# Save models
lstm_model.save('lstm_model.h5')
gru_model.save('gru_model.h5')

# Save the scaler using pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the SVM model using pickle
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
print("Predicted positions and models have been saved.")