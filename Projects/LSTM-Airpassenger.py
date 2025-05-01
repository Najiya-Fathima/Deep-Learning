# //
import numpy as np                         
import pandas as pd                       
import matplotlib.pyplot as plt

# //
data = pd.read_csv("AirPassengers.csv")
print(data.head())

# // 
data.describe()

# //
data.dtypes

# //

# Preprocess the data
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# //

# Plot to visualise dataset
plt.plot(data)
plt.title("Monthly passenger on airplane from 1949-01-01 to 1960-12-01")
plt.xlabel('Date')
plt.ylabel('Passenger')
plt.show()

# //

# Normalize the dataset
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# //

def create_sequence(data, time_step=12):
    x, y = [], []
    for i in range(len(data) - time_step):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

time_steps=12
X, y = create_sequence(scaled_data, time_step=time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X.shape, y.shape)


# //

# Build the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
from tensorflow.keras.layers import Input

model = Sequential()

model = Sequential()
model.add(Input(shape=(time_steps, 1)))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# //
model.summary()

# //

plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# //

# Predict on training data
y_pred = model.predict(X)

# Inverse transform to original scale
y_pred_inv = scaler.inverse_transform(y_pred)
y_actual_inv = scaler.inverse_transform(y.reshape(-1, 1))

# Plot
plt.figure(figsize=(10,6))
plt.plot(y_actual_inv, label='Actual Values')
plt.plot(y_pred_inv, label='Predicted Values')
plt.title('Actual vs Predicted Passenger Numbers')
plt.xlabel('Months')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# //

# Generate future predictions
last_data = scaled_data[-time_steps:].reshape(1, time_steps, 1)

num_future_points = 24  # 2 years

future_predictions = []

# Predict future values
current_batch = last_data.copy()
for i in range(num_future_points):
    current_pred = model.predict(current_batch)[0]
    future_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], 
                             [[current_pred]], 
                             axis=1)

# Convert future predictions to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions))

last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                            periods=num_future_points, 
                            freq='M')

# Plot original data with future predictions
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['#Passengers'], label='Historical Data')
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
plt.title('Air Passengers Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Passengers (thousands)')
plt.legend()
plt.grid(True)
plt.show()
