# Law School Admissions Predictor

A machine-learning web application that predicts your chances of admission to 193 ABA-accredited law schools based on LSAT, GPA, and other applicant data.

## Features

- Predicts admission probability for 193 law schools using gradient-boosted models
- Separate URM and non-URM models to reflect distinct admissions patterns
- Enhanced predictions that update in real time as you enter decisions from other schools
- School selectivity rankings derived from an Elo-style rating of historical LSAT/GPA medians
- Interaction features (LSAT x selectivity, GPA x selectivity, timing x selectivity) that capture how stats are valued differently at each tier
- Temporal weighting so recent admissions cycles carry more influence
- Clean, responsive web interface with school logos and T14 highlighting

## How It Works

The predictor uses four **HistGradientBoostingClassifier** models (scikit-learn):

| Model | Description |
|-------|-------------|
| Non-URM Baseline | Stats-only prediction (11 features) |
| Non-URM Enhanced | Adds cross-school decision signals (20 features) |
| URM Baseline | Stats-only, trained on URM applicant data |
| URM Enhanced | Adds cross-school decision signals for URM applicants |

When you enter only your LSAT, GPA, and other personal stats, the **baseline** model runs. Once you add acceptances, rejections, or waitlists from other schools, the app automatically switches to the **enhanced** model, which incorporates those outcomes using leave-one-out encoding to prevent data leakage.

Training uses exponential temporal decay (0.9^age) to down-weight older cycles and early stopping (500 iterations, 30-round patience) to prevent overfitting. Validation follows a temporal split: train on 2003-2021, validate on 2022-2023, and test on 2024-2025.

## Model Performance

| Model | Validation AUC | Test AUC |
|-------|:--------------:|:--------:|
| Non-URM Baseline | 0.906 | 0.878 |
| Non-URM Enhanced | 0.916 | 0.898 |
| URM Baseline | 0.849 | 0.842 |
| URM Enhanced | 0.883 | 0.877 |

Predictions are well-calibrated with a maximum calibration error of approximately 2.5 percentage points across deciles.

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (HistGradientBoostingClassifier), pandas, NumPy
- **Frontend:** HTML/CSS/JavaScript (Jinja2 templates)
- **Production server:** Gunicorn
- **Hosting:** Render

## Quick Start

```bash
git clone https://github.com/twgribble29/law-school-predictor.git
cd law-school-predictor
pip install -r requirements.txt
python app.py
```

Then visit [http://localhost:5001](http://localhost:5001).

## Deployment

This project is configured for one-click deployment on [Render](https://render.com). The `render.yaml` blueprint defines the web service, Python runtime, and Gunicorn start command. Connect your GitHub repo in the Render dashboard and it will build and deploy automatically.

## Data Source

All models are trained on self-reported admissions data from [LSD.Law](https://www.lsd.law), covering the 2003-2025 admissions cycles. The dataset contains approximately 100,000 application outcomes (~65,000 non-URM and ~12,000 URM) across 193 ABA-accredited law schools.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Ty Gribble** -- [github.com/twgribble29](https://github.com/twgribble29)
