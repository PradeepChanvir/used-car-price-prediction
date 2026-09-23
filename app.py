from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load dataset
df = pd.read_csv("cardekho_dataset.csv")

# Load trained ML model
model = joblib.load("best_car_price_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    # Default values for dropdowns
    selected_brand = ""
    selected_model = ""
    selected_car_name = ""

    if request.method == "POST":

        selected_brand = request.form["brand"]
        selected_model = request.form["model"]
        selected_car_name = request.form["car_name"]

        vehicle_age = int(request.form["vehicle_age"])
        km_driven = int(request.form["km_driven"])
        seller_type = request.form["seller_type"]
        fuel_type = request.form["fuel_type"]
        transmission_type = request.form["transmission_type"]
        mileage = float(request.form["mileage"])
        engine = int(request.form["engine"])
        max_power = float(request.form["max_power"])
        seats = int(request.form["seats"])

        input_data = pd.DataFrame({
            "car_name": [selected_car_name],
            "brand": [selected_brand],
            "model": [selected_model],
            "vehicle_age": [vehicle_age],
            "km_driven": [km_driven],
            "seller_type": [seller_type],
            "fuel_type": [fuel_type],
            "transmission_type": [transmission_type],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats]
        })

        predicted_price = model.predict(input_data)[0]
        prediction = f"₹{predicted_price:,.0f}"

    brands = sorted(df["brand"].unique())
    models = sorted(df["model"].unique())
    car_names = sorted(df["car_name"].unique())
    seller_types = sorted(df["seller_type"].unique())
    fuel_types = sorted(df["fuel_type"].unique())
    transmission_types = sorted(df["transmission_type"].unique())

    return render_template(
        "index.html",
        prediction=prediction,
        brands=brands,
        models=models,
        car_names=car_names,
        seller_types=seller_types,
        fuel_types=fuel_types,
        transmission_types=transmission_types,
        selected_brand=selected_brand,
        selected_model=selected_model,
        selected_car_name=selected_car_name
    )


if __name__ == "__main__":
    app.run(debug=True)
