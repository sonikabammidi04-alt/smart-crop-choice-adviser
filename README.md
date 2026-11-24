# 🌾 Smart Crop Choice Adviser using Generative AI

A Streamlit web app that recommends the best crop for a farmer's conditions, explains the choice in simple language, and reads the advice aloud.

**Live app:** _add your Streamlit link here_

## Features
- Inputs: soil type, season, region, temperature, rainfall, soil pH, market trend
- Scikit-learn pipeline (StandardScaler + OneHotEncoder + RandomForest) for crop classification
- Best crop plus **top-3 confidence scores**
- Farmer-friendly advisory:
  - **OpenAI** or **Gemini** (optional, bring your own key)
  - **Offline smart template** fallback, so it works with no API key
- Text-to-speech with **gTTS**: play in the app or download as MP3
- Advisory and audio in English, Hindi, Tamil, Telugu, Kannada and Marathi (AI mode)

## Tech stack
Python · Streamlit · Scikit-learn · Pandas · NumPy · gTTS · OpenAI / Gemini REST APIs

## How it works
1. A synthetic, agronomy-informed dataset (12 crops, noisy ranges for temperature, rainfall, pH, soil, season, region and market) is generated with NumPy and Pandas.
2. A Scikit-learn `Pipeline` preprocesses and trains a RandomForest classifier (cached on startup).
3. User inputs go through `predict_proba` to give the top-3 crops.
4. The result is turned into an advisory by an LLM, or by a rule-based template if no key is set.
5. gTTS converts the advisory to MP3 audio.

## Run locally
```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

## Optional API keys
Create `.streamlit/secrets.toml` (never commit it):
```toml
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "..."
```
Or paste a key in the app sidebar.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. (Optional) Under **Advanced settings → Secrets**, paste your API keys.
5. Click **Deploy** and copy your live link.

## Limitations and future work
- The training data is synthetic. Swap in a real dataset (e.g. Kaggle Crop Recommendation, or government agri data) for production use.
- gTTS needs internet access.
- Future: live weather and mandi price APIs, soil-test report upload, more crops and regions.

## Disclaimer
Advisory guidance only. Confirm with a local agriculture officer before sowing.
