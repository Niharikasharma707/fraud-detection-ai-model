import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re


nlp = spacy.load("en_core_web_sm")


model_name = "austinb/fraud_text_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


stopwords = nlp.Defaults.stop_words

FRAUD_THRESHOLD = 0.75

def classify_phrase_with_details(phrase):

    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True)


    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        fraud_score = probabilities[0][1].item()  
        predicted_class = int(fraud_score > FRAUD_THRESHOLD)  
    labels = ["non-fraud", "fraud"]

    reasons = {
        "non-fraud": "Phrase is unlikely to be fraudulent based on the current context.",
        "fraud": "High-risk phrase detected. There may be potential fraudulent behavior."
    }
    summaries = {
        "non-fraud": "This phrase appears safe with no significant risk of fraud.",
        "fraud": "This phrase indicates a high likelihood of fraudulent activity."
    }
    
    return {
        "phrase": phrase,
        "fraud_score": round(fraud_score, 2), 
        "label": labels[predicted_class], 
        "reason": reasons[labels[predicted_class]],  
        "summary": summaries[labels[predicted_class]]  
    }


def remove_stopwords(text):

    tokens = text.split()
    meaningful_words = [word for word in tokens if word.lower() not in stopwords]
    return meaningful_words

def split_into_phrases(text):
    text_with_periods = re.sub(r'\b(and|,|;)\b', '.', text)  
    segments = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.)\s', text_with_periods)
    segments = [seg.strip() for seg in segments if seg.strip()] 
    return segments


def classify_text_phrase_by_phrase(text):

    meaningful_phrases = split_into_phrases(text)

    results = []
    for phrase in meaningful_phrases:
        classification = classify_phrase_with_details(phrase) 
        results.append({
            "phrase": classification["phrase"],
            "fraud_score": classification["fraud_score"],
            "reason": classification["reason"],
            "summary": classification["summary"]
        })
    
    return results

sample_text = """
The customer named David ordered a luxury smartwatch (Rolex Watch Series 5) along with an unlimited data plan from Verizon on 15/09/2024. The billing address provided is incomplete, and the delivery address is a PO box located in a high-risk area (90210). The customer has ordered two similar high-value items within the last three days. The current order is being expedited despite the customer selecting standard shipping.
"""

result = classify_text_phrase_by_phrase(sample_text)

for entry in result:
    print(f"Phrase: {entry['phrase']}")
    print(f"Fraud Score: {entry['fraud_score']}")
    print(f"Reason: {entry['reason']}")
    print(f"Summary: {entry['summary']}")
    print("-" * 50)
