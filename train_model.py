import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Sample data - Replace with actual data
data = {
    'uneasyness_in_walking': [1, 7, 8, 2, 9, 4],
    'regular_dizziness': [2, 8, 7, 3, 9, 5],
    'vomition': [1, 7, 6, 2, 8, 4],
    'random_blacking_out': [1, 8, 9, 3, 7, 5],
    'parkinsons': [0, 1, 1, 0, 1, 0]  # 1 indicates Parkinson's, 0 indicates no Parkinson's
}

df = pd.DataFrame(data)

X = df.drop('parkinsons', axis=1)
y = df['parkinsons']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'parkinsons_model.joblib')

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
