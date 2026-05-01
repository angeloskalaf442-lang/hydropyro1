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


# =============================
# CONFIG
# =============================

st.set_page_config(
    page_title="HydroPyro AI",
    page_icon="🔥",
    layout="wide"
)

IMG_SIZE = (224, 224)
LSTM_SEQ_LEN = 30
LSTM_FEATURES = 8

TFLITE_MODEL = "hydropyro_3class_model.tflite"

HEADERS = {"User-Agent": "HydroPyro-MVP/3.0"}

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


# =============================
# DATA FUNCTIONS
# =============================

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

    url = (
        "https://api.open-meteo.com/v1/forecast"
        if is_future
        else "https://archive-api.open-meteo.com/v1/archive"
    )

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


# =============================
# MODEL LOADING
# =============================

@st.cache_resource
def load_tflite_model():
    """
    Tries to load TFLite model.
    If no compatible interpreter exists on Streamlit Cloud, it returns None safely.
    """
    if not os.path.exists(TFLITE_MODEL):
        return None, "TFLite file not found in repository"

    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=TFLITE_MODEL)
        interpreter.allocate_tensors()
        return interpreter, "TFLite / LiteRT model"

    except Exception as e1:
        try:
            from tflite_runtime.interpreter import Interpreter
            interpreter = Interpreter(model_path=TFLITE_MODEL)
            interpreter.allocate_tensors()
            return interpreter, "TFLite Runtime model"

        except Exception as e2:
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
                interpreter.allocate_tensors()
                return interpreter, "TensorFlow Lite model"

            except Exception as e3:
                return None, f"TFLite could not load: {e1} | {e2} | {e3}"


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
    return probs / probs.sum()


def run_prediction(img, weather, raw_weather_df):
    interpreter, model_status = load_tflite_model()

    if interpreter is not None:
        try:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            image_input = img[None, ...].astype(np.float32)
            weather_input = weather[None, ...].astype(np.float32)

            # Try normal input order
            try:
                interpreter.set_tensor(input_details[0]["index"], image_input)
                interpreter.set_tensor(input_details[1]["index"], weather_input)
            except Exception:
                # Try reverse input order
                interpreter.set_tensor(input_details[0]["index"], weather_input)
                interpreter.set_tensor(input_details[1]["index"], image_input)

            interpreter.invoke()

            probs = interpreter.get_tensor(output_details[0]["index"])[0]
            probs = np.array(probs, dtype=np.float32)
            probs = probs / probs.sum()

            return probs, model_status

        except Exception as e:
            probs = fallback_risk_prediction(raw_weather_df)
            return probs, f"Fallback MVP logic — TFLite inference error: {e}"

    probs = fallback_risk_prediction(raw_weather_df)
    return probs, f"Fallback MVP logic — {model_status}"


# =============================
# RISK LOGIC
# =============================

def classify_level(prob, predicted_class):
    if predicted_class == 0:
        return "LOW" if prob >= 0.75 else "UNCERTAIN"

    if prob >= 0.85:
        return "EXTREME"
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.45:
        return "MEDIUM"

    return "LOW"


def get_color(level, dominant_output):
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
    if gap >= 0.20:
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
        if recent_rain < previous_rain * 0.7:
            return "DECREASING"

        return "STABLE"

    if dominant_risk == "FIRE":
        recent_temp = recent["temperature_2m"].mean()
        previous_temp = previous["temperature_2m"].mean()
        recent_rh = recent["relative_humidity_2m"].mean()
        previous_rh = previous["relative_humidity_2m"].mean()

        if recent_temp > previous_temp + 2 and recent_rh < previous_rh:
            return "INCREASING"
        if recent_temp < previous_temp - 2 or recent_rh > previous_rh:
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
        if rain >= 30:
            return "HEAVY_RAINFALL_ANOMALY"
        return "NO_MAJOR_ANOMALY"

    if dominant_risk == "FIRE":
        if temp >= 38 and rh <= 25 and wind >= 35:
            return "EXTREME_FIRE_WEATHER_ANOMALY"
        if temp >= 32 and rh <= 35:
            return "FIRE_WEATHER_ANOMALY"
        return "NO_MAJOR_ANOMALY"

    return "NO_MAJOR_ANOMALY"


def generate_recommendation(dominant_risk, level):
    if dominant_risk == "NORMAL":
        return "No immediate disaster-prevention action required. Continue routine environmental monitoring."

    if dominant_risk == "FLOOD":
        if level in ["HIGH", "EXTREME"]:
            return "Increase monitoring of drainage systems, low-lying zones, riverbeds and vulnerable infrastructure. Prepare preventive civil-protection actions."
        if level == "MEDIUM":
            return "Monitor rainfall evolution and inspect flood-prone locations."
        return "Continue routine monitoring."

    if dominant_risk == "FIRE":
        if level in ["HIGH", "EXTREME"]:
            return "Increase surveillance of vegetation zones, monitor wind evolution, prepare firefighting resources and consider preventive restrictions on risky outdoor activity."
        if level == "MEDIUM":
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
        return f"Normal-risk output is supported by non-extreme observed conditions: {rain_sum:.1f} mm precipitation, {temp_max:.1f}°C maximum temperature, {rh_min:.1f}% minimum humidity, {wind_max:.1f} km/h maximum windspeed and {soil_mean:.2f} mean soil moisture."

    if dominant_risk == "FLOOD":
        return f"Flood risk is supported by accumulated precipitation of {rain_sum:.1f} mm, maximum windspeed of {wind_max:.1f} km/h and mean soil moisture of {soil_mean:.2f}."

    if dominant_risk == "FIRE":
        return f"Fire risk is supported by maximum temperature of {temp_max:.1f}°C, minimum relative humidity of {rh_min:.1f}%, maximum windspeed of {wind_max:.1f} km/h and accumulated precipitation of {rain_sum:.1f} mm."

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
    predicted_class = int(np.argmax(probs))
    dominant_output = CLASS_NAMES[predicted_class]
    dominant_probability = float(np.max(probs))

    level = classify_level(dominant_probability, predicted_class)

    output = {
        "location": {
            "name": city,
            "latitude": lat,
            "longitude": lon
        },
        "date": date_str,
        "risk_scores": {
            "normal_probability": round(float(probs[0]), 4),
            "fire_probability": round(float(probs[1]), 4),
            "flood_probability": round(float(probs[2]), 4)
        },
        "dominant_output": dominant_output,
        "dominant_probability": round(dominant_probability, 4),
        "risk_level": level,
        "priority_rank": priority_rank(level, dominant_output),
        "alert": generate_alert(dominant_output, level, city),
        "trend": detect_weather_trend(raw_weather_df, dominant_output),
        "anomaly": detect_anomaly(raw_weather_df, dominant_output),
        "confidence": confidence_score(probs),
        "recommended_action": generate_recommendation(dominant_output, level),
        "decision_justification": generate_justification(raw_weather_df, dominant_output),
        "model_status": model_status,
        "model_note": "HydroPyro MVP output. This is a decision-support estimate and must not replace official meteorological, hydrological or civil-protection assessment."
    }

    return output


def create_folium_map(lat, lon, city, output):
    color = get_color(output["risk_level"], output["dominant_output"])

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


# =============================
# UI
# =============================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 56px;
        font-weight: 900;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 24px;
        color: #555;
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
    st.write(
        """
        HydroPyro is an AI-driven platform for integrated wildfire and flood risk prediction,
        combining weather data, satellite imagery, geospatial indicators and machine-learning
        intelligence for environmental decision support.
        """
    )

    st.button("Request Pilot / Book Demo")

    c1, c2, c3 = st.columns(3)
    c1.metric("Outputs", "NORMAL / FIRE / FLOOD")
    c2.metric("Model", "TFLite-ready")
    c3.metric("Status", "MVP Online")

    st.header("Core Features")

    f1, f2, f3 = st.columns(3)
    f1.info("Live Prediction Tool")
    f2.info("Risk Dashboard")
    f3.info("Interactive Map")

    f4, f5, f6 = st.columns(3)
    f4.info("Alert System")
    f5.info("Decision Justification")
    f6.info("JSON Report")


elif page == "Live Prediction Tool":
    st.title("Live Prediction Tool")

    with st.form("prediction_form"):
        place = st.text_input("Exact city / location", "Thessaloniki")

        c1, c2, c3 = st.columns(3)
        day = c1.number_input("Day", min_value=1, max_value=31, value=1)
        month = c2.number_input("Month", min_value=1, max_value=12, value=5)
        year = c3.number_input("Year", min_value=2000, max_value=2035, value=2026)

        submitted = st.form_submit_button("Generate Prediction")

    if "result" not in st.session_state:
        st.session_state.result = None

    if submitted:
        try:
            date_obj = datetime(int(year), int(month), int(day))
            date_str = date_obj.strftime("%Y-%m-%d")

            with st.spinner("Running HydroPyro..."):
                lat, lon, city = get_coords(place)
                raw_weather_df = fetch_weather_dataframe(lat, lon, date_obj)
                weather = normalize_weather(raw_weather_df)
                img = fetch_nasa_image(lat, lon, date_obj)

                probs, model_status = run_prediction(img, weather, raw_weather_df)

                output = build_hydropyro_output(
                    city,
                    lat,
                    lon,
                    date_str,
                    probs,
                    raw_weather_df,
                    model_status
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

    if st.session_state.result:
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

        st.header("Recommended Action")
        st.write(output["recommended_action"])

        st.header("Decision Justification")
        st.write(output["decision_justification"])

        with st.expander("Weather / Data Table"):
            st.dataframe(raw_weather_df, width="stretch")

        with st.expander("Model Debug"):
            st.write("Files in repository:")
            st.write(os.listdir("."))
            st.write("TFLite exists:", os.path.exists(TFLITE_MODEL))

        st.header("Download Report")
        report_json = json.dumps(output, indent=2, ensure_ascii=False)

        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"hydropyro_report_{city.lower().replace(' ', '_')}.json",
            mime="application/json"
        )


elif page == "Business Model":
    st.title("Business Model")

    st.header("Pilot Project")
    st.write(
        """
        Pilot includes selected-region monitoring, wildfire/flood risk dashboard,
        map-based visualization, JSON reports and decision-support outputs.

        Suggested duration: 2–3 months.
        """
    )

    st.header("Pricing / Plans")

    p1, p2, p3 = st.columns(3)
    p1.info("Pilot Project")
    p2.info("Annual Subscription")
    p3.info("Premium Analytics")

    st.header("Target Customers")
    st.write(
        """
        Municipalities, civil protection agencies, utilities, insurers,
        environmental agencies and infrastructure operators.
        """
    )

    st.header("Use Cases")
    st.write(
        """
        - Wildfire risk for municipalities
        - Flood risk for cities
        - Infrastructure protection
        - Insurance risk documentation
        """
    )


elif page == "Technology & Transparency":
    st.title("Technology & Data Transparency")

    st.header("Technology")
    st.write(
        """
        HydroPyro combines satellite imagery, weather APIs, machine learning,
        geospatial visualization and environmental decision intelligence.
        """
    )

    st.header("Data Sources")
    st.write(
        """
        - Open-Meteo geocoding and weather/archive API
        - NASA Worldview / GIBS-style snapshot imagery
        - User-selected city/location and date
        """
    )

    st.header("Limitations")
    st.warning(
        """
        HydroPyro is an MVP decision-support tool.
        It must not replace official meteorological, hydrological, firefighting
        or civil-protection warnings.
        """
    )


elif page == "Partnerships":
    st.title("Partnerships")

    st.header("About Founder")
    st.write(
        """
        Angelos Kalafatas — Chemistry graduate with interests in environmental analysis,
        AI-based environmental decision support and climate-risk intelligence.
        """
    )

    st.header("Seeking Pilot Partners")
    st.info("Municipalities, research groups, accelerators, environmental agencies and infrastructure operators.")

    st.header("Case Study Examples")
    st.write(
        """
        - Thessaly Flood 2023
        - Rhodes Fire 2023
        - Attica Fire example
        """
    )


elif page == "Contact / Legal":
    st.title("Contact / Legal / Trust")

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
            st.error("Please provide GDPR consent.")
        else:
            st.success("Contact request captured in this session.")
            st.json(
                {
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
            )

    st.header("Disclaimer")
    st.warning(
        """
        HydroPyro is not an official emergency alert system.
        Always follow official authorities.
        """
    )
    
