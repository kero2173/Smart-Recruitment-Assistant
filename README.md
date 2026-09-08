# Smart Recruitment Assistant — Candidate Predictor

A small dark-themed Streamlit app that wraps your Task 5/6/7 pipeline:
it loads the already-trained `best_model.pkl` (Random Forest) and
`preprocessing.pkl` (scaler, mappings, training columns) and predicts
whether a candidate is likely to be looking for a job change.

No retraining happens here — this only reuses what your notebook already fit.

## How to run

1. Unzip this folder, make sure `best_model.pkl` and `preprocessing.pkl`
   sit next to `app.py`.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```
   streamlit run app.py
   ```
4. It opens in your browser at `http://localhost:8501`. Fill in a
   candidate's profile and click **Predict**.

## What it shows

- A prediction: "Likely looking for a job change" / "Likely to stay"
- The model's confidence (probability of class `1`)
- The exact processed input sent to the model (expandable, for debugging)

## Notes

- The model was trained on the HR Analytics dataset — the target means
  job-change intention, which this app frames as a recruitment-screening
  signal (not an automated hiring decision).
- Tried locally with two contrasting profiles:
  - Senior, stable-looking profile → prediction 0, P(job change) ≈ 0.02
  - Junior, high-flight-risk profile → prediction 1, P(job change) ≈ 0.59
