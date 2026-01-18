from flask import Flask, render_template, request
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from newspaper import Article
import torch
import re
app = Flask(__name__)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # 0 = low risk, 1 = high risk
)
model.eval()

def score_content(content):
    reasons = []
    risk_score = 0
    content = content.strip()

    if not content:
        return 0, ["No content provided"]

    # ----- Rule-based checks -----
    emotional_words = ["shocking", "unbelievable", "secret", "miracle", "exposed"]
    if any(word in content.lower() for word in emotional_words):
        risk_score += 3
        reasons.append("Contains emotionally manipulative words.")

    if re.search(r'\b[A-Z]{4,}\b', content):
        risk_score += 2
        reasons.append("Excessive capitalization detected, may indicate sensationalism.")

    if not any(word in content.lower() for word in ["source", "study", "report", "according to"]):
        risk_score += 2
        reasons.append("No credible source mentioned.")

    # ----- ML-based check -----
    inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        high_risk_prob = probs[0][1].item()  # probability of "high risk"
    
    ml_score = int(high_risk_prob * 3)  # scale 0-3
    risk_score += ml_score
    if high_risk_prob > 0.5:
        reasons.append(f"ML model flagged content as high risk ({high_risk_prob:.2f}).")

    # Clamp final score between 1-10
    risk_score = max(1, min(10, risk_score))

    return risk_score, reasons


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form.get('content', '').strip()

    if not user_input:
        return render_template('index.html', error="Please provide content or URL to analyze", content='')

    # URL Detection Logic
    if user_input.startswith(('http://', 'https://')):
        try:
            article = Article(user_input)
            article.download()
            article.parse()
            # We analyze the Title + Text for better context
            content_to_analyze = f"{article.title}. {article.text}"
        except Exception as e:
            return render_template('index.html', error=f"Could not read URL: {str(e)}", content=user_input)
    else:
        content_to_analyze = user_input

    score, reasons = score_content(content_to_analyze)
    return render_template('index.html', result={'score': score, 'reasons': reasons}, content=user_input)

if __name__ == '__main__':
    app.run(debug=True)