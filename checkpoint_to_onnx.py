from transformers import AutoModelForSequenceClassification, AutoTokenizer

local_path = "./checkpoint-4689"
model = AutoModelForSequenceClassification.from_pretrained(local_path)
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
print(type(model))
print(model.config.num_labels)
model.eval()
print(model.training)

## now we gonna tokenise the sample
question = "A traingle has sides 18, 24 and 30. find thear sa dsasa=fjsa"
## gibberish is fine because ok