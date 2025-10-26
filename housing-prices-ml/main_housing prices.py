import os
print("Current working directory:", os.getcwd())
print("Train exists:", os.path.exists(os.path.join("data", "train.csv")))
# src/model.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# --- Percorsi ---
train_path = os.path.join("data", "train.csv")
test_path = os.path.join("data", "test.csv")

# --- Caricamento dati ---
home_data = pd.read_csv(train_path)
y = home_data["SalePrice"]

# --- Selezione delle feature ---
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# --- Split dei dati per validazione ---
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# --- Modello Random Forest ---
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)

print(f"Validation MAE: {rf_val_mae:,.0f}")

# --- Addestramento su tutti i dati ---
rf_model_full = RandomForestRegressor(random_state=1)
rf_model_full.fit(X, y)

# --- Predizioni su test ---
test_data = pd.read_csv(test_path)
test_X = test_data[features]
test_preds = rf_model_full.predict(test_X)

# --- Output ---
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
print("File 'submission.csv' creato correttamente!")
