from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re


model_name = "austinb/fraud_text_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def classify_text_with_details(text_chunk):

    inputs = tokenizer(text_chunk, return_tensors="pt", truncation=True, padding=True)


    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
   
        probabilities = torch.softmax(logits, dim=-1)
        fraud_score = probabilities[0][1].item()  
        predicted_class = torch.argmax(logits, dim=-1).item()


    labels = ["non-fraud", "fraud"]
  
    reasons = {
        "non-fraud": "Low-risk transaction",
        "fraud": "Suspicious patterns detected"
    }
    summaries = {
        "non-fraud": "The text is deemed safe with no signs of fraud.",
        "fraud": "This part of the text indicates potential fraud."
    }
    
    return {
        "fraud_score": round(fraud_score, 2),  
        "label": labels[predicted_class],
        "reason": reasons[labels[predicted_class]],  
        "summary": summaries[labels[predicted_class]]  
    }

def classify_text_line_by_line(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)


    results = []
    for sentence in sentences:
        result = classify_text_with_details(sentence)
        results.append({"text": sentence, "result": result})
    
    return results
sample_text = """
The customer named Niharika placed an order for a smartphone device (iPhone) with the AT&T 4GB wireless package on 17/09/2024. 
The customer has not set up autopay, which may be a factor to consider. Zip code is 76767, and address is Signity Solutions. 
The user has received around 2-4 phones this week and the current order is already done.
"""

result = classify_text_line_by_line(sample_text)

for entry in result:
    print(f"Text: {entry['text']}")
    print(f"Fraud Score: {entry['result']['fraud_score']}")
    print(f"Label: {entry['result']['label']}")
    print(f"Reason: {entry['result']['reason']}")
    print(f"Summary: {entry['result']['summary']}")
    print("-" * 50)
