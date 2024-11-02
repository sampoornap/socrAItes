from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Read in the text file
with open("document.txt", "r") as file:
    document_text = file.read()

# Generate the summary (for long documents, you might need to split text)
summary = summarizer(document_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

print("Summary:")
print(summary)


from sklearn.feature_extraction.text import TfidfVectorizer

# Set up TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=30, stop_words='english', ngram_range=(1, 2))

# Fit and transform document text
tfidf_matrix = tfidf_vectorizer.fit_transform([document_text])

# Get top keywords
keywords = tfidf_vectorizer.get_feature_names_out()

print("Top Keywords:")
print(keywords)

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

entities = ner_pipeline(document_text)
unique_entities = {entity['word'] for entity in entities if entity['entity_group']}

print("Domain-Specific Entities:", unique_entities)
import json

data = {
    "summary": summary,
    "keywords": list(keywords),
    "entities": list(unique_entities)
}

with open("processed_document.json", "w") as f:
    json.dump(data, f, indent=4)

print("Data saved to processed_document.json")