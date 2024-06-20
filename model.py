import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv(r'C:\Users\Chandan\Desktop\ML PROJECT\Data\diabetes.csv')

# Separate features and target variable
X = df.drop('Outcome', axis=1).astype(float)
y = df['Outcome'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the model using pickle
filename = 'random_forest_model.sav'
pickle.dump(random_forest_model, open(filename, 'wb'))
