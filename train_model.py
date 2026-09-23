import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("cardekho_dataset.csv")
print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

before = len(df)
df = df.dropna()
print("\nRows removed due to missing values:", before - len(df))
print("Dataset shape after preprocessing:", df.shape)

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)

preprocessor = ColumnTransformer(
    transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nTraining Random Forest model...")
pipeline.fit(X_train, y_train)
print("Model training completed!")

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")
print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")
print("==========================================")

results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})
print("\nSample Predictions:")
print(results.to_string(index=False))
import joblib

joblib.dump(pipeline, "best_car_price_model.pkl")

print("\nModel saved successfully as best_car_price_model.pkl")
