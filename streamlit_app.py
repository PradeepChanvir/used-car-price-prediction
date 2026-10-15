import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AutoPrice - Used Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cardekho_dataset.csv"
MODEL_PATH = BASE_DIR / "best_car_price_model.pkl"
IMAGE_PATH = BASE_DIR / "images" / "hero_car.jpg"

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()

# ============================================================
# GLOBAL CSS
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f7f7fb;
}

/* Remove default Streamlit spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Navbar */
.navbar {
    background: white;
    border-radius: 18px;
    padding: 15px 28px;
    margin-bottom: 28px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 25px;
    font-weight: 800;
    color: #171717;
}

.logo span {
    color: #7c3aed;
}

.nav-text {
    color: #666;
    font-size: 14px;
    font-weight: 500;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #171329 0%, #31205e 55%, #6d28d9 100%);
    border-radius: 28px;
    padding: 55px;
    min-height: 440px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 45px rgba(79, 45, 140, 0.25);
}

.hero-content {
    width: 52%;
    position: relative;
    z-index: 3;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 8px 15px;
    border-radius: 30px;
    font-size: 13px;
    margin-bottom: 20px;
}

.hero h1 {
    font-size: 48px;
    line-height: 1.1;
    margin: 0 0 18px 0;
    font-weight: 800;
}

.hero h1 span {
    color: #c4b5fd;
}

.hero p {
    color: #ddd6fe;
    font-size: 17px;
    line-height: 1.7;
    margin-bottom: 25px;
}

.hero-image {
    position: absolute;
    right: -20px;
    bottom: 0;
    width: 55%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    opacity: 0.92;
    mask-image: linear-gradient(
        to right,
        transparent 0%,
        black 30%,
        black 100%
    );
    -webkit-mask-image: linear-gradient(
        to right,
        transparent 0%,
        black 30%,
        black 100%
    );
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        90deg,
        rgba(23,19,41,0.98) 0%,
        rgba(23,19,41,0.82) 35%,
        rgba(49,32,94,0.25) 70%,
        rgba(49,32,94,0.05) 100%
    );
    z-index: 1;
}

/* Section title */
.section-title {
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #18181b;
    margin: 45px 0 8px 0;
}

.section-subtitle {
    text-align: center;
    color: #71717a;
    margin-bottom: 28px;
}

/* Cards */
.info-card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.05);
    height: 100%;
    border: 1px solid #eee;
}

.info-icon {
    font-size: 30px;
    margin-bottom: 12px;
}

.info-card h3 {
    margin: 0 0 8px 0;
    color: #18181b;
    font-size: 20px;
}

.info-card p {
    color: #71717a;
    line-height: 1.6;
    font-size: 14px;
}

/* Prediction box */
.prediction-box {
    background: linear-gradient(135deg, #6d28d9, #9333ea);
    color: white;
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 12px 30px rgba(109,40,217,0.25);
}

.prediction-label {
    font-size: 14px;
    opacity: 0.85;
}

.prediction-price {
    font-size: 42px;
    font-weight: 800;
    margin-top: 8px;
}

/* Stats */
.stat-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #eee;
}

.stat-number {
    color: #6d28d9;
    font-size: 30px;
    font-weight: 800;
}

.stat-label {
    color: #71717a;
    font-size: 13px;
    margin-top: 5px;
}

/* Form */
.form-container {
    background: white;
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
    border: 1px solid #eee;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #9333ea);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 13px 25px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5b21b6, #7e22ce);
}

/* Footer */
.footer {
    text-align: center;
    color: #888;
    padding: 35px 0 10px 0;
    font-size: 13px;
}

</style>
""")

# ============================================================
# NAVBAR
# ============================================================

st.html("""
<div class="navbar">
    <div class="logo">Auto<span>Price</span> 🚘</div>
    <div class="nav-text">Machine Learning • Used Car Price Prediction</div>
</div>
""")

# ============================================================
# HERO SECTION
# ============================================================

if IMAGE_PATH.exists():

    import base64

    with open(IMAGE_PATH, "rb") as image_file:
        image_base64 = base64.b64encode(
            image_file.read()
        ).decode()

    image_src = f"data:image/jpeg;base64,{image_base64}"

    st.html(f"""
    <div class="hero">

        <div class="hero-overlay"></div>

        <img
            src="{image_src}"
            class="hero-image"
        >

        <div class="hero-content">

            <div class="hero-badge">
                ✨ Machine Learning Powered
            </div>

            <h1>
                Know Your Car's
                <span>Real Value.</span>
            </h1>

            <p>
                Predict the estimated selling price of a used car
                using vehicle specifications and a trained
                Random Forest machine learning model.
            </p>

        </div>

    </div>
    """)

else:

    st.warning(
        "Hero image not found. Please check images/hero_car.jpg"
    )

# ============================================================
# STATISTICS
# ============================================================

st.markdown(
    '<div class="section-title">Model at a Glance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Built and evaluated using real used-car data</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">15,411</div>
        <div class="stat-label">Dataset Records</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">93.77%</div>
        <div class="stat-label">R² Score</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">200</div>
        <div class="stat-label">Random Forest Trees</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">12+</div>
        <div class="stat-label">Vehicle Features</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Predict Your Car Price</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Enter the vehicle details below</div>',
    unsafe_allow_html=True
)

with st.container():

    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    brands = sorted(df["brand"].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    car_names = sorted(df["car_name"].dropna().unique())
    seller_types = sorted(df["seller_type"].dropna().unique())
    fuel_types = sorted(df["fuel_type"].dropna().unique())
    transmission_types = sorted(
        df["transmission_type"].dropna().unique()
    )

    with col1:
        brand = st.selectbox("Brand", brands)

        model_name = st.selectbox(
            "Model",
            models
        )

        car_name = st.selectbox(
            "Car Name",
            car_names
        )

        vehicle_age = st.number_input(
            "Vehicle Age (years)",
            min_value=0,
            max_value=30,
            value=5
        )

    with col2:

        km_driven = st.number_input(
            "Kilometres Driven",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000
        )

        seller_type = st.selectbox(
            "Seller Type",
            seller_types
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            fuel_types
        )

        transmission_type = st.selectbox(
            "Transmission",
            transmission_types
        )

    with col3:

        mileage = st.number_input(
            "Mileage",
            min_value=1.0,
            max_value=100.0,
            value=18.0,
            step=0.1
        )

        engine = st.number_input(
            "Engine (CC)",
            min_value=500,
            max_value=6000,
            value=1200,
            step=50
        )

        max_power = st.number_input(
            "Max Power (BHP)",
            min_value=20.0,
            max_value=1000.0,
            value=80.0,
            step=1.0
        )

        seats = st.number_input(
            "Seats",
            min_value=2,
            max_value=15,
            value=5
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        "🚀 Predict Estimated Selling Price"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame({
        "car_name": [car_name],
        "brand": [brand],
        "model": [model_name],
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

    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">
            Estimated Selling Price
        </div>
        <div class="prediction-price">
            ₹{predicted_price:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Three simple steps to estimate your car price</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🚘</div>
        <h3>1. Enter Details</h3>
        <p>
            Provide information such as brand, model,
            age, kilometres driven, fuel type,
            engine and other specifications.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🧠</div>
        <h3>2. ML Prediction</h3>
        <p>
            The trained Random Forest regression model
            processes the vehicle features and estimates
            its selling price.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">💰</div>
        <h3>3. Get Estimate</h3>
        <p>
            The application displays the estimated
            selling price instantly in Indian Rupees.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TECHNOLOGY
# ============================================================

st.markdown(
    '<div class="section-title">Technology Stack</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Technologies used to build the application</div>',
    unsafe_allow_html=True
)

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🐍</div>
        <h3>Python</h3>
        <p>Core programming language used for data processing and ML.</p>
    </div>
    """, unsafe_allow_html=True)

with tech2:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">📊</div>
        <h3>Scikit-learn</h3>
        <p>Machine learning library used to train the regression model.</p>
    </div>
    """, unsafe_allow_html=True)

with tech3:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">⚡</div>
        <h3>Streamlit</h3>
        <p>Framework used to create the interactive web application.</p>
    </div>
    """, unsafe_allow_html=True)

with tech4:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🌲</div>
        <h3>Random Forest</h3>
        <p>Regression algorithm used for used-car price prediction.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL PERFORMANCE AND ALGORITHM COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Comparison of Decision Tree, Support Vector Regression, '
    'and Random Forest on the test dataset'
    '</div>',
    unsafe_allow_html=True
)

# Load the actual results generated by train_model.py
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"

if COMPARISON_PATH.exists():

    comparison_df = pd.read_csv(COMPARISON_PATH)

    # Display detailed evaluation results
    st.subheader("Algorithm Comparison")

    display_df = comparison_df.copy()

    display_df["MAE"] = display_df["MAE"].apply(
        lambda value: f"₹{value:,.2f}"
    )

    display_df["RMSE"] = display_df["RMSE"].apply(
        lambda value: f"₹{value:,.2f}"
    )

    display_df["R2"] = display_df["R2"].apply(
        lambda value: f"{value * 100:.2f}%"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Highlight the selected model
    st.subheader("Best Performing Algorithm")

    best_model = comparison_df.loc[
        comparison_df["RMSE"].idxmin()
    ]

    st.success(
        f"🏆 {best_model['Model']} was selected as the best model "
        f"because it achieved the lowest RMSE of "
        f"₹{best_model['RMSE']:,.2f}."
    )

    # Compare R² scores
    st.subheader("R² Score Comparison")

    chart_df = comparison_df[
        ["Model", "R2"]
    ].set_index("Model")

    st.bar_chart(chart_df)

    # Explain the evaluation metrics
    with st.expander("Understand the evaluation metrics"):

        st.markdown("""
        **1. Mean Absolute Error (MAE)**

        MAE measures the average absolute difference between
        actual car prices and predicted prices. A lower MAE
        indicates more accurate predictions on average.

        **2. Root Mean Squared Error (RMSE)**

        RMSE measures prediction errors while penalizing
        larger errors more heavily. A lower RMSE indicates
        better performance under this metric.

        **3. R² Score**

        R² measures how well the model explains variations
        in the target values. A higher score is generally
        better. A negative R² means the model performed
        worse than a simple baseline that predicts the
        average target value on this test set.

        **Conclusion**

        Random Forest achieved the lowest RMSE and MAE,
        along with the highest R² score among the three
        tested models. Therefore, it is the preferred model
        for this experiment.
        """)

else:

    st.warning(
        "Model comparison results were not found. "
        "Run python train_model.py to generate model_comparison.csv."
    )
# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    AutoPrice — Used Car Price Prediction System<br>
    Developed using Machine Learning & Streamlit
</div>
""")
