from flask import Flask, render_template, request
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from requests_html import HTMLSession

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
    emotional_words = ["shocking", 
                       "unbelievable", 
                       "secret", 
                       "miracle", 
                       "exposed", 
                       "you won’t believe",
                        "what happens next",
                        "shocking truth",
                        "this will change everything",
                        "what nobody tells you",
                        "exposed",
                        "leaked",
                        "revealed",
                        "secret",
                        "hidden truth",
                        "must see",
                        "goes viral",
                        "breaking",
                        "disgusting",
                        "corrupt",
                        "evil",
                        "traitor",
                        "criminal",
                        "betrayal",
                        "destroying",
                        "lying",
                        "rigged",]
    
    if any(word in content.lower() for word in emotional_words):
        risk_score += 3
        reasons.append("Contains emotionally manipulative words.")

    if re.search(r'\b[A-Z]{4,}\b', content):
        risk_score += 2
        reasons.append("Excessive capitalization detected, may indicate sensationalism.")

    source_indicators = [
    "research",
    "data",
    "evidence",
    "findings",
    "statistics",
    "survey",
    "analysis",
    "published",
    "peer-reviewed",
    "journal",
    "official",
    "government",
    "experts say",
    "scientists say",
    "researchers say",
    "verified",
    "fact-checked",
    "documented",
    "case study",
    "white paper",
    "source", 
    "study", 
    "report", 
    "according to"
    ]

    if not any(word in content.lower() for word in source_indicators):
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
    if high_risk_prob >= 0.5:
        reasons.append(f"ML model flagged content as high risk ({high_risk_prob:.2f}).")
    else:
        reasons.append(f"ML model flagged content as low risk ({high_risk_prob:.2f}).")


    # Clamp final score between 1-10
    risk_score = max(1, min(10, risk_score))

    return risk_score, reasons, high_risk_prob


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
            session = HTMLSession()
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            response = session.get(user_input, headers=headers, timeout=10)
            title = response.html.find('h1', first=True)
            paragraphs = response.html.find('p')

            title_text = title.text if title else "No Title Found"
            content_text = " ".join([p.text for p in paragraphs])

            if not content_text.strip():
                return render_template('index.html', error="Could not extract text content from this URL.", content=user_input)
            
            # We analyze the Title + Text for better context
            content_to_analyze = f"{title_text}. {content_text}"

        except Exception as e:
            return render_template('index.html', error=f"Could not read URL: {str(e)}", content=user_input)
    else:
        content_to_analyze = user_input

    score, reasons, high_risk_prob = score_content(content_to_analyze)
    return render_template('index.html', result={'score': score, 'reasons': reasons, 'ml_percentage': int(high_risk_prob * 100)}, content=user_input)

if __name__ == '__main__':
    app.run(debug=True)