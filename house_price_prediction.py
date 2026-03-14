import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load Dataset
data = pd.read_csv("house_data.csv")

# Features and Target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully")
