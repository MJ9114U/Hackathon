# Information Trust Scorer

An AI-assisted tool that analyzes online content and provides a Trust Risk Score instead of a true/false label. It examines factors such as emotional language, missing sources, sensational headlines, and structural inconsistencies. The system explains why content is flagged, helping users make their own informed decisions.

## Features

- Analyze text content for trust signals
- Provide risk score based on multiple factors
- Explain flagged issues transparently
- Promote critical thinking

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the application: `python app.py`

Open your browser to `http://localhost:5000`

## Technologies

- Python
- Flask
- Transformers (for NLP)
- SpaCy