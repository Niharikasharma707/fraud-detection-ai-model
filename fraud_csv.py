import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import pandas as pd  # Import pandas for handling CSV files

# Load spaCy's NLP model
nlp = spacy.load("en_core_web_sm")

# Load the tokenizer and fraud detection model
model_name = "austinb/fraud_text_detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load stopwords
stopwords = nlp.Defaults.stop_words

# Define fraud score threshold
FRAUD_THRESHOLD = 0.75

# Function to classify a phrase as fraudulent or not
def classify_phrase_with_details(phrase):
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        fraud_score = probabilities[0][1].item()  # Fraud class score (assuming index 1 is fraud)
        predicted_class = int(fraud_score > FRAUD_THRESHOLD)  # Determine if fraud score exceeds threshold
    
    # Classification labels and reasons
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

# Function to split the text into phrases
def split_into_phrases(text):
    text_with_periods = re.sub(r'\b(and|,|;)\b', '.', text)  # Split based on connectors like "and"
    segments = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.)\s', text_with_periods)
    segments = [seg.strip() for seg in segments if seg.strip()]  # Clean empty segments
    return segments

# Function to classify each phrase in the text
def classify_text_phrase_by_phrase(text):
    meaningful_phrases = split_into_phrases(text)
    results = []
    for phrase in meaningful_phrases:
        classification = classify_phrase_with_details(phrase)  # Classify each phrase
        results.append({
            "phrase": classification["phrase"],
            "fraud_score": classification["fraud_score"],
            "reason": classification["reason"],
            "summary": classification["summary"]
        })
    return results

# Function to process CSV file and classify each row
def process_csv_file(file_path):
    # Load the CSV file using pandas
    df = pd.read_csv(file_path)

    # Iterate through each row of the CSV, assuming there is a 'text' column containing the content to analyze
    for index, row in df.iterrows():
        text = row['text']  # Assuming each row has a column named 'text'
        print(f"Processing row {index + 1}: {text}")
        
        # Classify the text phrase by phrase
        result = classify_text_phrase_by_phrase(text)
        
        # Print the results for each row
        for entry in result:
            print(f"Phrase: {entry['phrase']}")
            print(f"Fraud Score: {entry['fraud_score']}")
            print(f"Reason: {entry['reason']}")
            print(f"Summary: {entry['summary']}")
            print("-" * 50)

# Example usage: Load a CSV file and classify text in each row
csv_file_path = "C:/Users/Signity_Laptop/Downloads/fraudCSVBase.csv"
process_csv_file(csv_file_path)
