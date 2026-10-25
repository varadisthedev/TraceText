# TraceText

A Streamlit app that classifies text as **AI-generated** or **human-written**. Long input is split into chunks, each chunk is scored independently, and the results are ensembled into an overall AI/Human verdict.

## Features

- Paste any text (or load a Human / AI / Mixed sample) and get an ensembled AI vs. Human prediction
- Chunk-by-chunk breakdown, color-coded red (AI) / green (Human)
- Model performance metrics and dataset info on the same page

## Setup

```bash
git clone https://github.com/varadisthedev/TraceText.git
cd TraceText

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

> Requires scikit-learn with a version that can load `models/tracetext_model.joblib` (tested on scikit-learn 1.9.1 / Python 3.14). If no prebuilt wheel exists for your Python version, use Python 3.11–3.12.

## Run

```bash
streamlit run app.py
```

Then open the URL it prints (defaults to http://localhost:8501).

## Project structure

```
app.py              # Streamlit app
models/              # trained model (tracetext_model.joblib)
assets/              # hero banner image
notebook/            # training/evaluation notebook
.streamlit/          # theme config
```

## Dataset

Trained on [RAID](https://huggingface.co/datasets/liamdugan/raid) (Human vs. AI-generated text).

## Links

- Repo: https://github.com/varadisthedev/TraceText
- Dataset: https://huggingface.co/datasets/liamdugan/raid
