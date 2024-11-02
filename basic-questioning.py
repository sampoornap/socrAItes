import json
import random

with open("processed_document.json", "r") as file:
    data = json.load(file)

summary = data["summary"]
keywords = data["keywords"]

def fill_in_the_blank(text, keyword):
    return text.replace(keyword, "_____"), keyword

def generate_direct_question(concept):
    return f"What is {concept}?", concept

questions = []

# 1. Fill-in-the-blank questions from the summary
for keyword in keywords:
    if keyword in summary:
        question, answer = fill_in_the_blank(summary, keyword)
        questions.append({"question": question, "answer": answer})

# 2. Direct questions from keywords
for keyword in keywords:
    question, answer = generate_direct_question(keyword)
    questions.append({"question": question, "answer": answer})

for q in questions:
    print("Q:", q["question"])
    print("A:", q["answer"])


from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load T5 model fine-tuned on question generation
model_name = "valhalla/t5-small-qg-prepend"  # or any other fine-tuned T5 for QG
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_questions_t5(text):
    input_text = "generate question: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

complex_questions = []

sentences = summary.split(". ")

for sentence in sentences: 
    if sentence:
        question = generate_questions_t5(sentence)
        complex_questions.append({"question": question, "answer": sentence})

for q in complex_questions:
    print("Q:", q["question"])
    print("A:", q["answer"])