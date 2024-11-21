import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re


nlp = spacy.load("en_core_web_sm")

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
        "fraud_score": round(fraud_score, 2),  # Rounded confidence score
        "label": labels[predicted_class],  # Fraud or non-fraud label
        "reason": reasons[labels[predicted_class]],  # Reason for the classification
        "summary": summaries[labels[predicted_class]]  # Short summary
    }


def detect_dynamic_field(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Get named entities in the sentence
    if entities:
        return entities
    else:
        return [("General Information", "INFO")]  # Default if no entity is found


def validate_and_explain(classification):
    fraud_score = classification["fraud_score"]
    label = classification["label"]

    threshold = 0.7
    if label == "fraud" and fraud_score >= threshold:
        return classification
    else:
        return {
            "fraud_score": fraud_score,
            "label": "non-fraud",
            "reason": "Low confidence in fraud detection",
            "summary": "The text did not meet the criteria for fraud detection."
        }

def classify_words(text):

    words = text.split()
    
    # Classify each word
    results = []
    for word in words:
        dynamic_fields = detect_dynamic_field(word)  # Dynamically detect fields/entities
        classification = classify_text_with_details(word)  # Classify the word
        validated_classification = validate_and_explain(classification)
        results.append({
            "word": word,
            "detected_fields": dynamic_fields,
            "fraud_score": validated_classification["fraud_score"],
            "reason": validated_classification["reason"],
            "summary": validated_classification["summary"]
        })
    
    return results

# Example usage
sample_text = """
The customer named Niharika placed an order for a smartphone device (iPhone) with the AT&T 4GB wireless package on 17/09/2024 
The customer has not set up autopay which may be a factor to consider Zip code is 76767 and address is Signity Solutions 
The user has received around 2-4 phones this week and the current order is already done
"""

result = classify_words(sample_text)

# Print results
for entry in result:
    print(f"Word: {entry['word']}")
    print(f"Detected Fields: {entry['detected_fields']}")
    print(f"Fraud Score: {entry['fraud_score']}")
    print(f"Reason: {entry['reason']}")
    print(f"Summary: {entry['summary']}")
    print("-" * 50)
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

nlp = spacy.load("en_core_web_sm")


model_name = "austinb/fraud_text_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to classify a chunk of text as fraudulent or not
def classify_text_with_details(text_chunk):
    # Tokenize the input text
    inputs = tokenizer(text_chunk, return_tensors="pt", truncation=True, padding=True)

    # Perform prediction using the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Softmax to convert logits into probabilities
        probabilities = torch.softmax(logits, dim=-1)
        fraud_score = probabilities[0][1].item()  # Fraud class score (assuming index 1 is fraud)
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Map the output to the class names (assuming 0 = non-fraud, 1 = fraud)
    labels = ["non-fraud", "fraud"]
    
    # Define reasons and summary based on classification result
    reasons = {
        "non-fraud": "Low-risk transaction",
        "fraud": "Suspicious patterns detected"
    }
    summaries = {
        "non-fraud": "The text is deemed safe with no signs of fraud.",
        "fraud": "This part of the text indicates potential fraud."
    }
    
    return {
        "fraud_score": round(fraud_score, 2),  # Rounded confidence score
        "label": labels[predicted_class],  # Fraud or non-fraud label
        "reason": reasons[labels[predicted_class]],  # Reason for the classification
        "summary": summaries[labels[predicted_class]]  # Short summary
    }

def detect_dynamic_field(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Get named entities in the sentence
    if entities:
        return entities
    else:
        return [("General Information", "INFO")]  # Default if no entity is found

# Function to filter out invalid fraud detections and provide explanations
def validate_and_explain(classification):
    fraud_score = classification["fraud_score"]
    label = classification["label"]

    # Define criteria for a valid fraud detection
    # Example: Only consider fraud if the score is above a threshold
    threshold = 0.7
    if label == "fraud" and fraud_score >= threshold:
        return classification
    else:
        return {
            "fraud_score": fraud_score,
            "label": "non-fraud",
            "reason": "Low confidence in fraud detection",
            "summary": "The text did not meet the criteria for fraud detection."
        }

# Function to classify each word in the text
def classify_words(text):
    # Split the text into words based on spaces
    words = text.split()
    
    # Classify each word
    results = []
    for word in words:
        dynamic_fields = detect_dynamic_field(word)  # Dynamically detect fields/entities
        classification = classify_text_with_details(word)  # Classify the word
        validated_classification = validate_and_explain(classification)  # Validate the classification
        results.append({
            "word": word,
            "detected_fields": dynamic_fields,
            "fraud_score": validated_classification["fraud_score"],
            "reason": validated_classification["reason"],
            "summary": validated_classification["summary"]
        })
    
    return results

# Example usage
sample_text = """
The customer named Niharika placed an order for a smartphone device (iPhone) with the AT&T 4GB wireless package on 17/09/2024 
The customer has not set up autopay which may be a factor to consider Zip code is 76767 and address is Signity Solutions 
The user has received around 2-4 phones this week and the current order is already done
"""

result = classify_words(sample_text)

# Print results
for entry in result:
    print(f"Word: {entry['word']}")
    print(f"Detected Fields: {entry['detected_fields']}")
    print(f"Fraud Score: {entry['fraud_score']}")
    print(f"Reason: {entry['reason']}")
    print(f"Summary: {entry['summary']}")
    print("-" * 50)
