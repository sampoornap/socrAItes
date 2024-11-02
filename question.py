import yake

with open("document.txt", "r") as file:
    document_text = file.read()

# Use YAKE or any keyphrase extraction tool to extract main concepts
yake_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=10)
keyphrases = [phrase[0] for phrase in yake_extractor.extract_keywords(document_text)]

print("Main Concepts:", keyphrases)

from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "valhalla/t5-base-qg-hl"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_questions(text, concept):
    input_text = f"generate question: {text.replace(concept, '<hl>' + concept + '<hl>')}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

questions = []
for concept in keyphrases:
    question = generate_questions(document_text, concept)
    questions.append({"question": question, "concept": concept})

for q in questions:
    print("Q:", q["question"])