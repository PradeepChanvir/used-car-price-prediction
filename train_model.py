import pandas as pd
import numpy as np
import joblib
import time
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n==========================================")
print("USED CAR PRICE PREDICTION")
print("MODEL TRAINING AND COMPARISON")
print("==========================================")

df = pd.read_csv("cardekho_dataset.csv")

print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "selling_price" not in df.columns:
    raise ValueError(
        "The dataset must contain a 'selling_price' column."
    )

before = len(df)
df = df.dropna().copy()

print("Rows removed due to missing values:", before - len(df))
print("Dataset shape after preprocessing:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 4. DEFINE PREPROCESSING
# ==========================================

def create_preprocessor(scale_numeric=False):
    transformers = []

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        )

    if scale_numeric and numerical_features:
        transformers.append(
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"
    )


# ==========================================
# 5. DEFINE THREE MODELS
# ==========================================

models = {
    "Decision Tree": Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            DecisionTreeRegressor(
                random_state=42
            )
        )
    ]),

    "Support Vector Regression": Pipeline([
        (
            "preprocessor",
            create_preprocessor(scale_numeric=True)
        ),
        (
            "model",
            SVR(
                kernel="rbf",
                C=100,
                epsilon=0.1
            )
        )
    ]),

    "Random Forest": Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
        )
    ])
}


# ==========================================
# 6. TRAIN AND EVALUATE ALL MODELS
# ==========================================

comparison_results = []
trained_models = {}

for model_name, pipeline in models.items():

    print("\n------------------------------------------")
    print("Training:", model_name)
    print("------------------------------------------")

    start_time = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    comparison_results.append({
        "Model": model_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Training Time (seconds)": round(
            training_time, 2
        )
    })

    trained_models[model_name] = pipeline

    print(f"MAE: ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")


# ==========================================
# 7. SAVE COMPARISON RESULTS
# ==========================================

comparison_df = pd.DataFrame(comparison_results)

# Rank models using RMSE as the primary selection metric.
# MAE and R² are also displayed for interpretation.
comparison_df = comparison_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

comparison_df["Rank"] = range(
    1, len(comparison_df) + 1
)

comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)

with open("model_comparison.json", "w") as file:
    json.dump(
        comparison_results,
        file,
        indent=4
    )


# ==========================================
# 8. SELECT THE BEST MODEL
# ==========================================

best_model_name = comparison_df.iloc[0]["Model"]

print("\n==========================================")
print("MODEL COMPARISON RESULTS")
print("==========================================")

print(
    comparison_df[
        [
            "Rank",
            "Model",
            "MAE",
            "RMSE",
            "R2",
            "Training Time (seconds)"
        ]
    ].to_string(index=False)
)

print("\nBest model based on test RMSE:", best_model_name)


# ==========================================
# 9. SAVE THE BEST MODEL
# ==========================================

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "best_car_price_model.pkl"
)


with open("best_model.json", "w") as file:
    json.dump(
        {"best_model": best_model_name},
        file,
        indent=4
    )
print("\nSaved files:")
print("- best_car_price_model.pkl")
print("- model_comparison.csv")
print("- model_comparison.json")
print("- best_model.json")

print("\nModel comparison and training completed successfully!")
