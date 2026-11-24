"""
Smart Crop Choice Adviser using Generative AI
---------------------------------------------
Streamlit app that:
  1. Takes soil / season / weather / pH / region / market inputs
  2. Trains a Scikit-learn pipeline (OneHotEncoder + StandardScaler + RandomForest)
  3. Shows the best crop + top-3 confidence scores
  4. Writes a farmer-friendly advisory (OpenAI / Gemini, or an offline template)
  5. Converts the advisory to audio with gTTS (playable + downloadable)
"""

import re
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from gtts import gTTS
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# 1. DOMAIN KNOWLEDGE (agronomy profiles used to build the training data)
# ----------------------------------------------------------------------------
SOILS = ["Alluvial", "Black", "Red", "Sandy", "Loamy", "Clay"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
REGIONS = ["North", "South", "East", "West", "Central"]
MARKETS = ["Rising", "Stable", "Falling"]

# temp = deg C, rain = seasonal rainfall in mm, market = P(Rising, Stable, Falling)
CROP_PROFILES = {
    "Rice": dict(temp=(22, 32), rain=(1000, 2000), ph=(5.5, 7.0),
                 soils=["Alluvial", "Clay", "Loamy"], seasons=["Kharif"],
                 regions=["East", "South"], market=(0.3, 0.5, 0.2)),
    "Wheat": dict(temp=(10, 25), rain=(400, 900), ph=(6.0, 7.5),
                  soils=["Alluvial", "Loamy", "Black"], seasons=["Rabi"],
                  regions=["North", "Central"], market=(0.3, 0.5, 0.2)),
    "Maize": dict(temp=(18, 30), rain=(500, 1000), ph=(5.8, 7.2),
                  soils=["Loamy", "Red", "Alluvial"], seasons=["Kharif"],
                  regions=["Central", "South"], market=(0.4, 0.4, 0.2)),
    "Cotton": dict(temp=(22, 35), rain=(600, 1100), ph=(6.0, 8.0),
                   soils=["Black", "Alluvial"], seasons=["Kharif"],
                   regions=["West", "Central"], market=(0.3, 0.4, 0.3)),
    "Sugarcane": dict(temp=(24, 36), rain=(1100, 2000), ph=(6.0, 8.0),
                      soils=["Alluvial", "Black", "Loamy"], seasons=["Zaid", "Kharif"],
                      regions=["West", "North", "South"], market=(0.3, 0.5, 0.2)),
    "Groundnut": dict(temp=(22, 33), rain=(500, 1000), ph=(5.8, 7.0),
                      soils=["Sandy", "Red", "Loamy"], seasons=["Kharif"],
                      regions=["West", "South"], market=(0.4, 0.4, 0.2)),
    "Pearl Millet (Bajra)": dict(temp=(25, 36), rain=(300, 600), ph=(6.0, 8.0),
                                 soils=["Sandy", "Red"], seasons=["Kharif"],
                                 regions=["West", "North"], market=(0.5, 0.3, 0.2)),
    "Chickpea": dict(temp=(15, 28), rain=(300, 700), ph=(6.0, 8.0),
                     soils=["Black", "Loamy", "Sandy"], seasons=["Rabi"],
                     regions=["Central", "North", "West"], market=(0.5, 0.3, 0.2)),
    "Soybean": dict(temp=(20, 32), rain=(600, 1100), ph=(6.0, 7.5),
                    soils=["Black", "Loamy"], seasons=["Kharif"],
                    regions=["Central"], market=(0.4, 0.4, 0.2)),
    "Mustard": dict(temp=(10, 25), rain=(250, 600), ph=(6.0, 7.5),
                    soils=["Alluvial", "Loamy", "Sandy"], seasons=["Rabi"],
                    regions=["North", "East"], market=(0.4, 0.4, 0.2)),
    "Tomato": dict(temp=(18, 30), rain=(400, 900), ph=(6.0, 7.0),
                   soils=["Loamy", "Red", "Sandy"], seasons=["Zaid", "Rabi"],
                   regions=["South", "West"], market=(0.3, 0.3, 0.4)),
    "Watermelon": dict(temp=(24, 35), rain=(300, 600), ph=(6.0, 7.0),
                       soils=["Sandy", "Loamy"], seasons=["Zaid"],
                       regions=["North", "West"], market=(0.4, 0.4, 0.2)),
}

NUM_COLS = ["temperature", "rainfall", "ph"]
CAT_COLS = ["soil", "season", "region", "market"]

LANGUAGES = {"English": "en", "Hindi": "hi", "Tamil": "ta",
             "Telugu": "te", "Kannada": "kn", "Marathi": "mr"}


# ----------------------------------------------------------------------------
# 2. DATA + MODEL (Pandas, NumPy, Scikit-learn)
# ----------------------------------------------------------------------------
def _pick(rng, preferred, everything, n, prob):
    """Choose from `preferred` with probability `prob`, else from any value."""
    good = rng.choice(preferred, n)
    anyv = rng.choice(everything, n)
    return np.where(rng.random(n) < prob, good, anyv)


def generate_dataset(n_per_crop=400, seed=42):
    """Build a synthetic but agronomy-informed training set with noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for crop, p in CROP_PROFILES.items():
        n = n_per_crop
        (t_lo, t_hi), (r_lo, r_hi), (h_lo, h_hi) = p["temp"], p["rain"], p["ph"]
        frames.append(pd.DataFrame({
            # uniform inside the ideal range, with 10% spill-over for realism
            "temperature": rng.uniform(t_lo - 0.1 * (t_hi - t_lo), t_hi + 0.1 * (t_hi - t_lo), n),
            "rainfall": rng.uniform(r_lo - 0.1 * (r_hi - r_lo), r_hi + 0.1 * (r_hi - r_lo), n),
            "ph": rng.uniform(h_lo - 0.1 * (h_hi - h_lo), h_hi + 0.1 * (h_hi - h_lo), n),
            "soil": _pick(rng, p["soils"], SOILS, n, 0.85),
            "season": _pick(rng, p["seasons"], SEASONS, n, 0.90),
            "region": _pick(rng, p["regions"], REGIONS, n, 0.75),
            "market": rng.choice(MARKETS, n, p=p["market"]),
            "crop": crop,
        }))
    # shuffle rows
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


@st.cache_resource(show_spinner="Training the crop model...")
def train_model():
    """Train once per server session; returns (pipeline, hold-out accuracy)."""
    df = generate_dataset()
    X, y = df[NUM_COLS + CAT_COLS], df["crop"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    pipeline = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipeline.predict(X_te))
    return pipeline, float(acc)


def predict_top3(model, inputs):
    """Return [(crop, probability), ...] for the 3 most likely crops."""
    row = pd.DataFrame([inputs])[NUM_COLS + CAT_COLS]
    proba = model.predict_proba(row)[0]
    top_idx = np.argsort(proba)[::-1][:3]
    return [(str(model.classes_[i]), float(proba[i])) for i in top_idx]


# ----------------------------------------------------------------------------
# 3. ADVISORY GENERATION (LLM optional, template fallback always works)
# ----------------------------------------------------------------------------
def fit_notes(crop, inp):
    """Rule-based comparison of the farmer's inputs vs. the crop's ideal profile."""
    p = CROP_PROFILES[crop]
    notes = []

    t_lo, t_hi = p["temp"]
    if t_lo <= inp["temperature"] <= t_hi:
        notes.append(f"Your average temperature of {inp['temperature']:.0f} degrees Celsius is "
                     f"well inside the {t_lo} to {t_hi} degree range that {crop} likes.")
    elif inp["temperature"] > t_hi:
        notes.append(f"It is a bit hotter than ideal for {crop}, so plan mulching and timely irrigation.")
    else:
        notes.append(f"It is a bit cooler than ideal for {crop}, so protect young plants from cold.")

    r_lo, r_hi = p["rain"]
    if r_lo <= inp["rainfall"] <= r_hi:
        notes.append(f"Rainfall of about {inp['rainfall']:.0f} millimetres suits this crop.")
    elif inp["rainfall"] < r_lo:
        notes.append("Rainfall is lower than ideal, so keep irrigation ready, such as drip or sprinkler.")
    else:
        notes.append("Rainfall is higher than ideal, so make sure the field has good drainage.")

    h_lo, h_hi = p["ph"]
    if h_lo <= inp["ph"] <= h_hi:
        notes.append(f"Your soil pH of {inp['ph']:.1f} is in the healthy range.")
    elif inp["ph"] < h_lo:
        notes.append("Your soil is on the acidic side. Adding agricultural lime can help.")
    else:
        notes.append("Your soil is on the alkaline side. Adding gypsum or compost can help.")

    if inp["soil"] in p["soils"]:
        notes.append(f"{inp['soil']} soil is a good match for {crop}.")
    else:
        notes.append(f"{inp['soil']} soil is not the usual choice, so add compost and "
                     "manage water carefully.")

    if inp["market"] == "Rising":
        notes.append("Market prices are rising, which is a good sign for your income.")
    elif inp["market"] == "Falling":
        notes.append("Market prices are falling, so consider selling through a contract or storing the produce.")
    else:
        notes.append("Market prices are steady, so you can plan with confidence.")
    return notes


def template_advisory(top3, inp):
    """Offline fallback: a natural, farmer-friendly explanation built from rules."""
    best, best_p = top3[0]
    others = [f"{c} ({p * 100:.0f} percent)" for c, p in top3[1:]]
    intro = (f"For the {inp['season']} season in the {inp['region']} region, "
             f"{best} is the best choice for you, with about {best_p * 100:.0f} percent confidence.")
    why = " ".join(fit_notes(best, inp))
    backup = f"Good backup options are {' and '.join(others)}." if others else ""
    tip = ("Please also check with your local agriculture office for seed varieties and "
           "current prices before you sow.")
    return "\n\n".join(part for part in [intro, why, backup, tip] if part)


SYSTEM_PROMPT = ("You are a friendly agriculture adviser speaking to small farmers. "
                 "Use simple words, short sentences, and no markdown or bullet symbols.")


def build_prompt(top3, inp, language):
    top_txt = ", ".join(f"{c} ({p * 100:.0f}%)" for c, p in top3)
    return (
        f"A machine-learning model recommends these crops: {top_txt}.\n"
        f"Farmer inputs: soil={inp['soil']}, season={inp['season']}, region={inp['region']}, "
        f"temperature={inp['temperature']} C, rainfall={inp['rainfall']} mm, "
        f"soil pH={inp['ph']}, market trend={inp['market']}.\n"
        f"Write a short advisory (under 150 words) in {language}: say which crop to grow and why, "
        "mention one or two practical tips, and name the backup crops. Do not use markdown."
    )


def call_openai(key, model, prompt):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0.5,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def call_gemini(key, model, prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_advisory(provider, key, model_name, language, top3, inp):
    """Returns (text, language_code_used, warning_or_None). Never raises."""
    if provider != "Offline template" and key:
        try:
            prompt = build_prompt(top3, inp, language)
            text = (call_openai if provider == "OpenAI" else call_gemini)(key, model_name, prompt)
            if text:
                return text, LANGUAGES[language], None
        except Exception as exc:  # network error, bad key, wrong model name...
            return (template_advisory(top3, inp), "en",
                    f"AI call failed ({type(exc).__name__}). Showing the offline advisory instead.")
    warn = None
    if provider != "Offline template" and not key:
        warn = "No API key provided. Showing the offline advisory instead."
    elif language != "English":
        warn = "The offline advisory is English only. Choose an AI provider for other languages."
    return template_advisory(top3, inp), "en", warn


# ----------------------------------------------------------------------------
# 4. TEXT-TO-SPEECH (gTTS)
# ----------------------------------------------------------------------------
def clean_for_speech(text):
    """Strip markdown symbols and extra whitespace so the voice sounds natural."""
    text = re.sub(r"[*#_`>]", "", text)
    return re.sub(r"\s+", " ", text).strip()[:1800]


def make_audio(text, lang_code):
    """Convert text to MP3 bytes. gTTS needs an internet connection."""
    buf = BytesIO()
    gTTS(text=text, lang=lang_code, slow=False).write_to_fp(buf)
    return buf.getvalue()


def get_secret(name):
    """Read a key from Streamlit secrets without crashing if none are set."""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# 5. STREAMLIT UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Smart Crop Choice Adviser", page_icon="🌾", layout="centered")
st.title("🌾 Smart Crop Choice Adviser")
st.caption("ML-powered crop recommendation with generative AI advice and voice output.")

model, model_acc = train_model()

# ---- Sidebar: advisory settings ----
with st.sidebar:
    st.header("⚙️ Advisory settings")
    provider = st.radio("Advisory engine", ["Offline template", "OpenAI", "Gemini"])
    api_key, model_name = "", ""
    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API key", type="password",
                                value=get_secret("OPENAI_API_KEY"))
        model_name = st.text_input("Model name", value="gpt-4o-mini")
    elif provider == "Gemini":
        api_key = st.text_input("Gemini API key", type="password",
                                value=get_secret("GEMINI_API_KEY"))
        model_name = st.text_input("Model name", value="gemini-2.5-flash")
    language = st.selectbox("Advisory & audio language", list(LANGUAGES))
    st.divider()
    st.caption(f"Model: RandomForest pipeline · hold-out accuracy {model_acc * 100:.1f}% "
               "(on synthetic agronomy data)")

# ---- Input form ----
with st.form("crop_form"):
    c1, c2 = st.columns(2)
    soil = c1.selectbox("Soil type", SOILS)
    season = c2.selectbox("Season", SEASONS)
    region = c1.selectbox("Location / region", REGIONS)
    market = c2.selectbox("Market trend", MARKETS)
    temperature = st.slider("Average temperature (°C)", 5, 45, 28)
    rainfall = st.slider("Seasonal rainfall (mm)", 0, 3000, 800, step=50)
    ph = st.slider("Soil pH", 4.0, 9.0, 6.5, step=0.1)
    submitted = st.form_submit_button("🌱 Recommend my crop", use_container_width=True)

if submitted:
    inputs = {"soil": soil, "season": season, "region": region, "market": market,
              "temperature": float(temperature), "rainfall": float(rainfall), "ph": float(ph)}
    top3 = predict_top3(model, inputs)
    with st.spinner("Writing your advisory..."):
        advisory, lang_used, warning = generate_advisory(
            provider, api_key, model_name, language, top3, inputs)
    spoken_top = ", ".join(f"{c} at {p * 100:.0f} percent" for c, p in top3)
    speech = clean_for_speech(
        f"Recommended crop: {top3[0][0]}. Top three options are {spoken_top}. {advisory}")
    st.session_state["result"] = {"top3": top3, "advisory": advisory, "warning": warning,
                                  "speech": speech, "lang": lang_used, "audio": None}

res = st.session_state.get("result")
if res:
    top3 = res["top3"]
    st.divider()
    st.subheader("✅ Recommendation")
    st.metric("Best crop", top3[0][0], f"{top3[0][1] * 100:.1f}% confidence", delta_color="off")

    chart_df = pd.DataFrame({"Confidence (%)": [round(p * 100, 1) for _, p in top3]},
                            index=[c for c, _ in top3])
    st.bar_chart(chart_df)

    st.subheader("🗣️ Farmer-friendly advisory")
    if res["warning"]:
        st.info(res["warning"])
    st.write(res["advisory"])

    st.subheader("🔊 Listen to it")
    if st.button("Generate audio"):
        with st.spinner("Creating audio..."):
            try:
                res["audio"] = make_audio(res["speech"], res["lang"])
            except Exception as exc:
                st.error(f"Audio generation failed ({type(exc).__name__}). "
                         "gTTS needs an internet connection.")
    if res["audio"]:
        st.audio(res["audio"], format="audio/mp3")
        st.download_button("⬇️ Download MP3", data=res["audio"],
                           file_name="crop_advice.mp3", mime="audio/mpeg")

    st.caption("This tool gives guidance only. Please confirm with your local agriculture "
               "officer before sowing.")
