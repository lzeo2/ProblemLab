from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

path = "AMC_2024_Intermediate.pdf"

# Load questions from JSON
with open(f"{path}.json", "r", encoding="utf-8") as file:
    pretokenisedquestions = json.load(file)

# Remove unnecessary whitespace and newlines
cleaned_questions = [
    " ".join(question.split())
    for question in pretokenisedquestions
]

print(cleaned_questions)
print(len(cleaned_questions))
inputlist = []
for question in cleaned_questions:
    inputs = tokenizer(question, truncation=True, max_length = 1024, padding=False)
    inputlist.append(dict(inputs))
temp = json.dumps(inputlist)

with open(f"{path}tokenised.json", "w", encoding="utf-8") as file:
    file.write(temp)
    

