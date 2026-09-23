import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(120, 100, 255, 0.18), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(0, 200, 255, 0.16), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(180, 80, 255, 0.14), transparent 35%),
        linear-gradient(135deg, #eef4ff 0%, #f7f2ff 50%, #edfaff 100%);
}

/* Hide Streamlit default elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main container */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%);
    border-radius: 28px;
    padding: 48px 50px;
    color: white;
    box-shadow: 0 20px 50px rgba(79, 70, 229, 0.30);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
    right: -80px;
    top: -100px;
}

.hero::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: rgba(0,220,255,0.10);
    left: -70px;
    bottom: -100px;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin: 0 0 12px 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 18px;
    line-height: 1.6;
    opacity: 0.92;
    max-width: 700px;
    margin-bottom: 22px;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 50px;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
}

/* Information cards */
.info-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(255,255,255,0.95);
    border-radius: 20px;
    padding: 24px 18px;
    text-align: center;
    min-height: 145px;
    box-shadow: 0 10px 30px rgba(50, 60, 100, 0.10);
}

.info-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.info-title {
    font-size: 17px;
    font-weight: 800;
    color: #172033;
    margin-bottom: 7px;
}

.info-text {
    font-size: 13px;
    color: #687386;
}

/* Section */
.section-card {
    background: rgba(255,255,255,0.90);
    border-radius: 24px;
    padding: 30px;
    margin-top: 28px;
    margin-bottom: 18px;
    box-shadow: 0 10px 35px rgba(50, 60, 100, 0.10);
    border: 1px solid rgba(255,255,255,0.95);
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #182033;
    margin-bottom: 5px;
}

.section-description {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Input labels */
label {
    font-weight: 600 !important;
    color: #263246 !important;
}

/* Streamlit inputs */
.stSelectbox > div > div,
.stNumberInput > div > div {
    border-radius: 12px !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 16px;
    padding: 16px 20px;
    font-size: 18px;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    box-shadow: 0 10px 25px rgba(79,70,229,0.30);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(79,70,229,0.40);
}

/* Prediction result */
.result-card {
    background: linear-gradient(135deg, #111827, #1e293b);
    border-radius: 24px;
    padding: 35px;
    text-align: center;
    margin-top: 28px;
    box-shadow: 0 15px 40px rgba(15,23,42,0.25);
}

.result-label {
    color: #cbd5e1;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
}

.result-price {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin: 0;
}

.result-note {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 10px;
}

/* Footer */
.custom-footer {
    text-align: center;
    padding: 30px 10px 10px 10px;
    color: #64748b;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cardekho_dataset.csv")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_car_price_model.pkl")


df = load_data()
model = load_model()


# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-title">🚗 Used Car Price Predictor</div>
        <div class="hero-subtitle">
            Estimate the market value of a used car using Machine Learning.
            Enter the vehicle details below and get an instant price prediction.
        </div>
        <div class="hero-badge">
            ⚡ Random Forest Regression &nbsp; • &nbsp; 📊 Data-Driven Prediction
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# INFORMATION CARDS
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🤖</div>
        <div class="info-title">ML Powered</div>
        <div class="info-text">Random Forest Regression</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">📈</div>
        <div class="info-title">Model Performance</div>
        <div class="info-text">R² Score: 93.77%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🚘</div>
        <div class="info-title">Multiple Features</div>
        <div class="info-text">12 vehicle attributes</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">⚡</div>
        <div class="info-title">Instant Result</div>
        <div class="info-text">Fast price estimation</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# CAR INFORMATION SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="section-card">
    <div class="section-title">🚘 Car Information</div>
    <div class="section-description">
        Select the basic details of the vehicle.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    brands = sorted(df["brand"].dropna().unique())
    selected_brand = st.selectbox(
        "🏷️ Brand",
        brands
    )

with col2:
    models = sorted(df["model"].dropna().unique())
    selected_model = st.selectbox(
        "🚘 Model",
        models
    )

with col3:
    car_names = sorted(df["car_name"].dropna().unique())
    selected_car_name = st.selectbox(
        "🚗 Car Name",
        car_names
    )

col1, col2, col3 = st.columns(3)

with col1:
    vehicle_age = st.number_input(
        "📅 Vehicle Age (Years)",
        min_value=0,
        max_value=30,
        value=5,
        step=1
    )

with col2:
    km_driven = st.number_input(
        "🛣️ Kilometers Driven",
        min_value=0,
        max_value=1000000,
        value=50000,
        step=1000
    )

with col3:
    seller_types = sorted(df["seller_type"].dropna().unique())
    seller_type = st.selectbox(
        "👤 Seller Type",
        seller_types
    )


# ---------------------------------------------------------
# TECHNICAL SPECIFICATIONS
# ---------------------------------------------------------
st.markdown("""
<div class="section-card">
    <div class="section-title">⚙️ Technical Specifications</div>
    <div class="section-description">
        Enter the technical specifications of the vehicle.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    fuel_types = sorted(df["fuel_type"].dropna().unique())
    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        fuel_types
    )

with col2:
    transmission_types = sorted(
        df["transmission_type"].dropna().unique()
    )
    transmission_type = st.selectbox(
        "⚙️ Transmission",
        transmission_types
    )

with col3:
    mileage = st.number_input(
        "📊 Mileage (km/l)",
        min_value=0.0,
        max_value=100.0,
        value=18.0,
        step=0.1
    )

col1, col2, col3 = st.columns(3)

with col1:
    engine = st.number_input(
        "🔧 Engine (CC)",
        min_value=500,
        max_value=10000,
        value=1200,
        step=100
    )

with col2:
    max_power = st.number_input(
        "⚡ Max Power (bhp)",
        min_value=20.0,
        max_value=2000.0,
        value=80.0,
        step=5.0
    )

with col3:
    seats = st.number_input(
        "💺 Seats",
        min_value=2,
        max_value=15,
        value=5,
        step=1
    )


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔮 Predict Used Car Price"
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if predict_button:

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

    try:
        predicted_price = model.predict(input_data)[0]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Used Car Market Value</div>
            <div class="result-price">₹{predicted_price:,.0f}</div>
            <div class="result-note">
                This estimate is generated using the trained Random Forest regression model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="custom-footer">
    🚗 Used Car Price Prediction System &nbsp; • &nbsp;
    Machine Learning Application
    <br>
    Built using Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)
