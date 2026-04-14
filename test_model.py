import joblib
import pandas as pd

m = joblib.load('model.pkl')
print('Model type:', type(m))

row = pd.DataFrame([{
    'area (sq ft)': 7420,
    'bedrooms': 4,
    'bathrooms': 2,
    'stories': 3,
    'guestroom': 0,
    'basement': 0,
    'airconditioning': 1,
    'prefarea': 1,
    'parking': 2,
    'furnishingstatus_semi-furnished': 0,
    'furnishingstatus_unfurnished': 0
}])

print('Prediction:', m.predict(row))