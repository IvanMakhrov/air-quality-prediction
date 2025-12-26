# Air Quality Prediction 🌍

**Predict European Air Quality Index (AQI)** using weather, geographic, and
emission data. This project implements an MLOps pipeline for training,
evaluation, and logging of tabular regression models.

---

## 📦 Setup

### Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Installation

```bash
git clone https://github.com/your-username/air-quality-prediction.git
cd air-quality-prediction

# Create virtual environment and install deps
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

uv sync --all-extras
pre-commit install

air-quality-prediction train
air-quality-prediction train model=tabtransformer
```
