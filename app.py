import os
import json
from io import BytesIO
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import folium
from streamlit_folium import st_folium

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


IMG_SIZE = (224, 224)
LSTM_SEQ_LEN = 30
LSTM_FEATURES = 8
MODEL_OUT = "hydropyro_3class_model.keras"
HEADERS = {"User-Agent": "HydroPyro-MVP/2.0"}

CLASS_NAMES = {
    0: "NORMAL",
    1: "FIRE",
    2: "FLOOD"
}

WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "precipitation",
    "windspeed_10m",
    "et0_fao_evapotranspiration",
    "surface_pressure",
    "soil_moisture_0_to_7cm"
]

WEATHER_MIN_MAX = {
    "temperature_2m": (-10, 50),
    "relative_humidity_2m": (0, 100),
    "dewpoint_2m": (-10, 30),
    "precipitation": (0, 50),
    "windspeed_10m": (0, 100),
    "et0_fao_evapotranspiration": (0, 15),
    "surface_pressure": (950, 1050),
    "soil_moisture_0_to_7cm": (0, 1)
}


def get_coords(query: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 1, "language": "en", "format": "json"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json().get("results")
    if not data:
        raise Exception("Location not found.")
    return float(data[0]["latitude"]), float(data[0]["longitude"]), data[0]["name"]


def fetch_weather_dataframe(lat, lon, date_obj):
    now = datetime.now()
    is_future = date_obj.date() > now.date()

    url = "https://api.open-meteo.com/v1/forecast" if is_future else "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_COLS),
        "timezone": "UTC"
    }

    if is_future:
        params["forecast_days"] = 3
    else:
        params["start_date"] = (date_obj - timedelta(days=2)).strftime("%Y-%m-%d")
        params["end_date"] = date_obj.strftime("%Y-%m-%d")

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=25)
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
        if not hourly:
            return pd.DataFrame(columns=WEATHER_COLS)

        df = pd.DataFrame(hourly)

        for col in WEATHER_COLS:
            if col not in df.columns:
                df[col] = np.nan

        return df[WEATHER_COLS].ffill().bfill()

    except Exception:
        return pd.DataFrame(columns=WEATHER_COLS)


def normalize_weather(df):
    if df.empty:
        return np.zeros((LSTM_SEQ_LEN, LSTM_FEATURES), dtype=np.float32)

    norm = df.copy()

    for col in WEATHER_COLS:
        mi, ma = WEATHER_MIN_MAX[col]
        norm[col] = (norm[col] - mi) / (ma - mi + 1e-7)

    data = np.nan_to_num(norm.values.astype(np.float32))

    if len(data) < LSTM_SEQ_LEN:
        pad = np.zeros((LSTM_SEQ_LEN - len(data), LSTM_FEATURES), dtype=np.float32)
        data = np.vstack([pad, data])

    return data[-LSTM_SEQ_LEN:]


def fetch_nasa_image(lat, lon, date_obj):
    q_date = min(date_obj, datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"
    box = 0.2

    params = {
        "REQUEST": "GetSnapshot",
        "TIME": q_date,
        "BBOX": f"{lon-box},{lat-box},{lon+box},{lat+box}",
        "CRS": "EPSG:4326",
        "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "FORMAT": "image/jpeg",
        "WIDTH": 224,
        "HEIGHT": 224
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=25)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return np.array(img.resize(IMG_SIZE), dtype=np.float32) / 255.0

    except Exception:
        return np.zeros((224, 224, 3), dtype=np.float32)


def build_hybrid_model():
    if not TF_AVAILABLE:
        return None

    img_in = layers.Input(shape=(224, 224, 3), name="image_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal")(img_in)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(96, (3, 3), activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    seq_in = layers.Input(shape=(LSTM_SEQ_LEN, LSTM_FEATURES), name="sensor_input")

    y = layers.LSTM(64, return_sequences=True)(seq_in)
    y = layers.LSTM(32)(y)

    combined = layers.Concatenate()([x, y])

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.35)(z)
    z = layers.Dense(48, activation="relu")(z)
    z = layers.Dropout(0.25)(z)

    out = layers.Dense(3, activation="softmax", name="output")(z)

    model = models.Model(inputs=[img_in, seq_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


@st.cache_resource
def load_hydropyro_model():
    if TF_AVAILABLE and os.path.exists(MODEL_OUT):
        return tf.keras.models.load_model(MODEL_OUT)
    return None


def fallback_risk_prediction(raw_weather_df):
    if raw_weather_df.empty:
        return np.array([0.70, 0.15, 0.15], dtype=np.float32)

    temp_max = raw_weather_df["temperature_2m"].max()
    rh_min = raw_weather_df["relative_humidity_2m"].min()
    rain_sum = raw_weather_df["precipitation"].sum()
    wind_max = raw_weather_df["windspeed_10m"].max()
    soil_mean = raw_weather_df["soil_moisture_0_to_7cm"].mean()

    fire = 0.05
    flood = 0.05

    if temp_max >= 32:
        fire += 0.25
    if temp_max >= 38:
        fire += 0.20
    if rh_min <= 35:
        fire += 0.20
    if rh_min <= 25:
        fire += 0.15
    if wind_max >= 30:
        fire += 0.15
    if rain_sum <= 2:
        fire += 0.10

    if rain_sum >= 20:
        flood += 0.30
    if rain_sum >= 50:
        flood += 0.25
    if soil_mean >= 0.55:
        flood += 0.15
    if soil_mean >= 0.75:
        flood += 0.20
    if wind_max >= 45:
        flood += 0.10

    fire = min(fire, 0.95)
    flood = min(flood, 0.95)
    normal = max(0.05, 1.0 - max(fire, flood))

    probs = np.array([normal, fire, flood], dtype=np.float32)
    probs = probs / probs.sum()

    return probs


def run_prediction(img, weather, raw_weather_df):
    model = load_hydropyro_model()

    if model is not None:
        probs = model.predict(
            {
                "image_input": img[None, ...],
                "sensor_input": weather[None, ...]
            },
            verbose=0
        )[0]
        return probs, "CNN-LSTM Keras model"

    probs = fallback_risk_prediction(raw_weather_df)
    return probs, "Fallback MVP logic — upload hydropyro_3class_model.keras for real model inference"


def classify_level(prob, predicted_class):
    if predicted_class == 0:
        if prob >= 0.75:
            return "LOW"
        return "UNCERTAIN"

    if prob >= 0.85:
        return "EXTREME"
    elif prob >= 0.70:
        return "HIGH"
    elif prob >= 0.45:
        return "MEDIUM"

    return "LOW"


def get_color(level, dominant_output=None):
    if dominant_output == "NORMAL":
        return "green"

    return {
        "LOW": "green",
        "UNCERTAIN": "blue",
        "MEDIUM": "orange",
        "HIGH": "red",
        "EXTREME": "darkred"
    }.get(level, "blue")


def confidence_score(probs):
    sorted_probs = sorted(probs, reverse=True)
    gap = sorted_probs[0] - sorted_probs[1]

    if gap >= 0.45:
        return "HIGH"
    elif gap >= 0.20:
        return "MEDIUM"

    return "LOW"


def detect_weather_trend(df, dominant_risk):
    if df.empty or len(df) < 8:
        return "INSUFFICIENT_DATA"

    recent = df.tail(8)
    previous = df.iloc[:-8]

    if previous.empty:
        return "INSUFFICIENT_DATA"

    if dominant_risk == "FLOOD":
        recent_rain = recent["precipitation"].mean()
        previous_rain = previous["precipitation"].mean()

        if recent_rain > previous_rain * 1.5 and recent_rain > 1:
            return "INCREASING"
        elif recent_rain < previous_rain * 0.7:
            return "DECREASING"

        return "STABLE"

    if dominant_risk == "FIRE":
        recent_temp = recent["temperature_2m"].mean()
        previous_temp = previous["temperature_2m"].mean()
        recent_rh = recent["relative_humidity_2m"].mean()
        previous_rh = previous["relative_humidity_2m"].mean()

        if recent_temp > previous_temp + 2 and recent_rh < previous_rh:
            return "INCREASING"
        elif recent_temp < previous_temp - 2 or recent_rh > previous_rh:
            return "DECREASING"

        return "STABLE"

    return "STABLE"


def detect_anomaly(df, dominant_risk):
    if df.empty:
        return "UNKNOWN"

    rain = df["precipitation"].sum()
    temp = df["temperature_2m"].max()
    rh = df["relative_humidity_2m"].min()
    wind = df["windspeed_10m"].max()
    soil = df["soil_moisture_0_to_7cm"].mean()

    if dominant_risk == "FLOOD":
        if rain >= 80 or soil >= 0.75:
            return "EXTREME_RAINFALL_OR_SOIL_MOISTURE_ANOMALY"
        elif rain >= 30:
            return "HEAVY_RAINFALL_ANOMALY"

        return "NO_MAJOR_ANOMALY"

    if dominant_risk == "FIRE":
        if temp >= 38 and rh <= 25 and wind >= 35:
            return "EXTREME_FIRE_WEATHER_ANOMALY"
        elif temp >= 32 and rh <= 35:
            return "FIRE_WEATHER_ANOMALY"

        return "NO_MAJOR_ANOMALY"

    return "NO_MAJOR_ANOMALY"


def generate_recommendation(dominant_risk, level):
    if dominant_risk == "NORMAL":
        return "No immediate disaster-prevention action required. Continue routine environmental monitoring."

    if dominant_risk == "FLOOD":
        if level in ["HIGH", "EXTREME"]:
            return (
                "Increase monitoring of drainage systems, low-lying zones, riverbeds and vulnerable infrastructure. "
                "Prepare preventive civil-protection actions."
            )
        elif level == "MEDIUM":
            return "Monitor rainfall evolution and inspect flood-prone locations."

        return "Continue routine monitoring."

    if dominant_risk == "FIRE":
        if level in ["HIGH", "EXTREME"]:
            return (
                "Increase surveillance of vegetation zones, monitor wind evolution, prepare firefighting resources "
                "and consider preventive restrictions on risky outdoor activity."
            )
        elif level == "MEDIUM":
            return "Monitor temperature, humidity and wind conditions. Prepare preventive checks."

        return "Continue routine monitoring."

    return "Continue monitoring."


def generate_justification(df, dominant_risk):
    if df.empty:
        return "Risk output generated with limited weather data availability."

    temp_max = df["temperature_2m"].max()
    rh_min = df["relative_humidity_2m"].min()
    rain_sum = df["precipitation"].sum()
    wind_max = df["windspeed_10m"].max()
    soil_mean = df["soil_moisture_0_to_7cm"].mean()

    if dominant_risk == "NORMAL":
        return (
            f"Normal-risk output is supported by non-extreme observed conditions: "
            f"{rain_sum:.1f} mm accumulated precipitation, {temp_max:.1f}°C maximum temperature, "
            f"{rh_min:.1f}% minimum relative humidity, {wind_max:.1f} km/h maximum windspeed "
            f"and {soil_mean:.2f} mean near-surface soil moisture."
        )

    if dominant_risk == "FLOOD":
        return (
            f"Flood risk is supported by accumulated precipitation of {rain_sum:.1f} mm, "
            f"maximum windspeed of {wind_max:.1f} km/h and mean near-surface soil moisture of {soil_mean:.2f}."
        )

    if dominant_risk == "FIRE":
        return (
            f"Fire risk is supported by maximum temperature of {temp_max:.1f}°C, "
            f"minimum relative humidity of {rh_min:.1f}%, maximum windspeed of {wind_max:.1f} km/h "
            f"and accumulated precipitation of {rain_sum:.1f} mm."
        )

    return "Risk justification unavailable."


def generate_alert(dominant_risk, level, city):
    if dominant_risk == "NORMAL":
        return f"✅ NORMAL CONDITIONS in {city}. Routine monitoring is sufficient."

    if level == "EXTREME":
        return f"🚨 EXTREME {dominant_risk} RISK in {city}. Immediate preparedness review recommended."

    if level == "HIGH":
        return f"⚠️ HIGH {dominant_risk} RISK in {city}. Preventive monitoring recommended."

    if level == "MEDIUM":
        return f"⚠️ MEDIUM {dominant_risk} RISK in {city}. Conditions should be monitored."

    return f"✅ LOW {dominant_risk} RISK in {city}. Routine monitoring is sufficient."


def priority_rank(level, dominant_risk):
    if dominant_risk == "NORMAL":
        return 5

    return {
        "EXTREME": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
        "UNCERTAIN": 4
    }.get(level, 4)


def build_hydropyro_output(city, lat, lon, date_str, probs, raw_weather_df, model_status):
    normal_prob = float(probs[0])
    fire_prob = float(probs[1])
    flood_prob = float(probs[2])

    predicted_class = int(np.argmax(probs))
    dominant_output = CLASS_NAMES[predicted_class]
    dominant_probability = float(np.max(probs))

    level = classify_level(dominant_probability, predicted_class)
    trend = detect_weather_trend(raw_weather_df, dominant_output)
    anomaly = detect_anomaly(raw_weather_df, dominant_output)
    confidence = confidence_score(probs)

    output = {
        "location": {
            "name": city,
            "latitude": lat,
            "longitude": lon
        },
        "date": date_str,
        "risk_scores": {
            "normal_probability": round(normal_prob, 4),
            "fire_probability": round(fire_prob, 4),
            "flood_probability": round(flood_prob, 4)
        },
        "dominant_output": dominant_output,
        "dominant_probability": round(dominant_probability, 4),
        "risk_level": level,
        "priority_rank": priority_rank(level, dominant_output),
        "alert": generate_alert(dominant_output, level, city),
        "trend": trend,
        "anomaly": anomaly,
        "confidence": confidence,
        "recommended_action": generate_recommendation(dominant_output, level),
        "decision_justification": generate_justification(raw_weather_df, dominant_output),
        "model_status": model_status,
        "model_note": (
            "HydroPyro MVP output. This is a decision-support estimate and must not replace "
            "official meteorological, hydrological or civil-protection assessment."
        )
    }

    return output


def create_folium_map(lat, lon, city, output):
    level = output["risk_level"]
    dominant = output["dominant_output"]
    color = get_color(level, dominant)

    m = folium.Map(location=[lat, lon], zoom_start=9)

    popup = f"""
    <b>{city}</b><br>
    Dominant output: {output["dominant_output"]}<br>
    Normal probability: {output["risk_scores"]["normal_probability"] * 100:.1f}%<br>
    Fire probability: {output["risk_scores"]["fire_probability"] * 100:.1f}%<br>
    Flood probability: {output["risk_scores"]["flood_probability"] * 100:.1f}%<br>
    Risk level: {output["risk_level"]}<br>
    Trend: {output["trend"]}<br>
    Priority rank: {output["priority_rank"]}
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=20,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.65,
        popup=popup
    ).add_to(m)

    return m


st.set_page_config(
    page_title="HydroPyro AI",
    page_icon="🔥",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 54px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 24px;
        color: #555;
    }
    .section-card {
        padding: 22px;
        border-radius: 18px;
        background-color: #f7f7f8;
        margin-bottom: 18px;
    }
    .small-muted {
        color: #666;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("🔥 HydroPyro")

page = st.sidebar.radio(
    "Navigation",
    [
        "Landing Page",
        "Live Prediction Tool",
        "Business Model",
        "Technology & Transparency",
        "Partnerships",
        "Contact / Legal"
    ]
)


if page == "Landing Page":
    st.markdown('<div class="main-title">🔥 HydroPyro</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI wildfire & flood risk intelligence</div>', unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            HydroPyro is an AI-driven platform for integrated wildfire and flood risk prediction,
            combining weather data, satellite imagery, geospatial indicators and machine-learning
            intelligence for environmental decision support.
            """
        )
        st.button("Request Pilot / Book Demo")

    with col2:
        st.metric("Outputs", "NORMAL / FIRE / FLOOD")
        st.metric("Use", "Decision Support")
        st.metric("Status", "MVP Demo")

    st.divider()

    st.header("Core Website Features")

    c1, c2, c3 = st.columns(3)
    c1.info("Live Prediction Tool")
    c2.info("Risk Dashboard")
    c3.info("Interactive Map")

    c4, c5, c6 = st.columns(3)
    c4.info("Decision Justification")
    c5.info("JSON Report")
    c6.info("Pilot Requests")


elif page == "Live Prediction Tool":
    st.title("Live Prediction Tool")
    st.caption("Enter exact city/location and date, then generate a HydroPyro risk estimate.")

    with st.form("prediction_form"):
        place = st.text_input("Exact city / location", "Thessaloniki")

        col1, col2, col3 = st.columns(3)

        day = col1.number_input("Day", min_value=1, max_value=31, value=1)
        month = col2.number_input("Month", min_value=1, max_value=12, value=5)
        year = col3.number_input("Year", min_value=2000, max_value=2035, value=2026)

        submitted = st.form_submit_button("Generate Prediction")
if "result" not in st.session_state:
    st.session_state.result = None

if submitted:
    try:
        date_obj = datetime(int(year), int(month), int(day))
        date_str = date_obj.strftime("%Y-%m-%d")

        with st.spinner("Fetching coordinates, weather data, NASA image and running HydroPyro..."):
            lat, lon, city = get_coords(place)
            raw_weather_df = fetch_weather_dataframe(lat, lon, date_obj)
            weather = normalize_weather(raw_weather_df)
            img = fetch_nasa_image(lat, lon, date_obj)
            probs, model_status = run_prediction(img, weather, raw_weather_df)

            output = build_hydropyro_output(
                city=city,
                lat=lat,
                lon=lon,
                date_str=date_str,
                probs=probs,
                raw_weather_df=raw_weather_df,
                model_status=model_status
            )

            st.session_state.result = {
                "output": output,
                "lat": lat,
                "lon": lon,
                "city": city,
                "raw_weather_df": raw_weather_df
            }

    except Exception as e:
        st.error(f"Prediction failed: {e}")

if st.session_state.result is not None:
    output = st.session_state.result["output"]
    lat = st.session_state.result["lat"]
    lon = st.session_state.result["lon"]
    city = st.session_state.result["city"]
    raw_weather_df = st.session_state.result["raw_weather_df"]

    st.success(output["alert"])
    st.caption(f"Model status: {output['model_status']}")

    st.header("Risk Dashboard")

    m1, m2, m3 = st.columns(3)
    m1.metric("NORMAL", f"{output['risk_scores']['normal_probability'] * 100:.1f}%")
    m2.metric("FIRE", f"{output['risk_scores']['fire_probability'] * 100:.1f}%")
    m3.metric("FLOOD", f"{output['risk_scores']['flood_probability'] * 100:.1f}%")

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Dominant", output["dominant_output"])
    d2.metric("Risk level", output["risk_level"])
    d3.metric("Confidence", output["confidence"])
    d4.metric("Trend", output["trend"])
    d5.metric("Anomaly", output["anomaly"])
    d6.metric("Priority", output["priority_rank"])

    st.header("Interactive Map")
    fmap = create_folium_map(lat, lon, city, output)
    st_folium(fmap, width=1100, height=520)

    st.header("Alert System")
    st.warning(output["alert"])

    st.header("Recommended Action")
    st.write(output["recommended_action"])

    st.header("Decision Justification")
    st.write(output["decision_justification"])

    with st.expander("Weather / Data Table"):
        st.dataframe(raw_weather_df, use_container_width=True)

    st.header("Download Report")
    report_json = json.dumps(output, indent=2, ensure_ascii=False)

    st.download_button(
        "Download JSON Report",
        data=report_json,
        file_name=f"hydropyro_report_{city.lower().replace(' ', '_')}.json",
        mime="application/json"
    )

     st.info("PDF report can be added later. Current MVP exports JSON.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


elif page == "Business Model":
    st.title("Business Model")

    st.header("Pilot Project Section")

    st.markdown(
        """
        **Pilot Project includes:**
        - Selected municipality / region monitoring
        - Wildfire and flood risk dashboard
        - Location/date predictions
        - Map-based visualization
        - JSON reporting
        - Pilot feedback loop

        **Suggested duration:** 2–3 months  
        **For:** municipalities, civil protection, utilities, insurers, environmental agencies and infrastructure operators.
        """
    )

    st.button("Request Pilot")

    st.header("Pricing / Plans")

    col1, col2, col3 = st.columns(3)

    col1.info("Pilot Project\n\nEntry pilot for selected region")
    col2.info("Annual Subscription\n\nContinuous risk dashboard access")
    col3.info("Premium Analytics\n\nAdvanced reports, integrations and custom analysis")

    st.header("Target Customers")

    st.write(
        """
        Municipalities, civil protection agencies, utilities, insurers, environmental agencies,
        infrastructure operators and regional climate-risk teams.
        """
    )

    st.header("Use Cases")

    st.write(
        """
        - Wildfire risk for municipalities
        - Flood risk for cities
        - Infrastructure protection
        - Insurance risk documentation
        - Environmental early-warning support
        """
    )

    st.header("How It Works")

    st.write(
        """
        1. Input location and date  
        2. Fetch weather / environmental / satellite data  
        3. Run AI prediction  
        4. Produce map, report and recommended action  
        """
    )


elif page == "Technology & Transparency":
    st.title("Technology & Data Transparency")

    st.header("Technology Section")

    st.write(
        """
        HydroPyro uses satellite imagery, weather APIs, CNN-LSTM / machine-learning architecture,
        geospatial processing and environmental risk intelligence.
        """
    )

    st.header("Data Sources")

    st.write(
        """
        - Open-Meteo geocoding and weather/archive API
        - NASA Worldview Snapshots / GIBS-style satellite imagery
        - User-selected city/location and date
        """
    )

    st.header("Update Frequency")

    st.write(
        """
        Weather and forecast availability depends on API access and date selection.
        Historical data uses archive endpoints; future dates use forecast endpoints.
        """
    )

    st.header("Model Limitations")

    st.warning(
        """
        HydroPyro MVP is a decision-support tool. It must not replace official meteorological,
        hydrological, firefighting or civil-protection warnings.
        """
    )


elif page == "Partnerships":
    st.title("Partnerships & Credibility")

    st.header("About Founder / Team")

    st.write(
        """
        **Angelos Kalafatas**  
        Chemistry graduate with interests in environmental analysis, analytical chemistry,
        AI-based environmental decision support and climate-risk intelligence.
        """
    )

    st.header("Social Links")

    st.write(
        """
        - GitHub: add your GitHub link  
        - LinkedIn: add your LinkedIn link  
        - Email: add your email  
        - Instagram/X: optional  
        """
    )

    st.header("Mobile App Section")

    st.info("Google Play: Coming Soon — no fake download link until an actual app exists.")

    st.header("Partnership Section")

    st.write(
        """
        HydroPyro is seeking pilot partners:
        - municipalities
        - research groups
        - accelerators
        - environmental agencies
        - infrastructure operators
        """
    )

    st.header("Case Study / Demo Examples")

    st.write(
        """
        Demo examples to include later:
        - Thessaly Flood 2023
        - Rhodes Fire 2023
        - Attica Fire example
        """
    )

    st.header("Testimonials / Logos")

    st.info("Seeking pilot partners — logos/testimonials will be added after real pilots.")


elif page == "Contact / Legal":
    st.title("Contact / Legal / Trust")

    st.header("Contact Form")

    with st.form("contact_form"):
        name = st.text_input("Name")
        organization = st.text_input("Organization")
        email = st.text_input("Email")
        role = st.text_input("Role")
        region = st.text_input("Region")
        use_case = st.text_area("Use case")
        message = st.text_area("Message")

        gdpr = st.checkbox("I consent to be contacted about HydroPyro pilot opportunities.")

        sent = st.form_submit_button("Submit Contact Request")

    if sent:
        if not gdpr:
            st.error("Please provide GDPR consent before submitting.")
        else:
            contact_payload = {
                "name": name,
                "organization": organization,
                "email": email,
                "role": role,
                "region": region,
                "use_case": use_case,
                "message": message,
                "gdpr_consent": gdpr,
                "submitted_at": datetime.now().isoformat()
            }

            st.success("Contact request captured in this session.")
            st.json(contact_payload)

    st.header("Disclaimer")

    st.warning(
        """
        HydroPyro is an MVP decision-support platform. It is not an official emergency alert system.
        Always follow official civil-protection, meteorological, hydrological and firefighting authorities.
        """
    )

    st.header("Privacy Policy")

    st.write(
        """
        Contact-form data should only be used to respond to pilot or partnership requests.
        A full privacy policy should be added before commercial launch.
        """
    )

    st.header("Terms of Use")

    st.write(
        """
        HydroPyro outputs are provided for informational and decision-support purposes only.
        They do not constitute official emergency, insurance, legal or engineering advice.
        """
    )
