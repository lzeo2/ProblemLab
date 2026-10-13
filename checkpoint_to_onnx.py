from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. Load fine-tuned ModernBERT
local_path = "./checkpoint-4689"

model = AutoModelForSequenceClassification.from_pretrained(local_path)
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

model.eval()

print("Model:", type(model).__name__)
print("Number of labels:", model.config.num_labels)

# 2. Create example inputs
question = "A triangle has sides 18, 24 and 30. Find its area."

inputs = tokenizer(
    question,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("Input IDs:", input_ids.shape)
print("Attention mask:", attention_mask.shape)

# 3. Prepare export arguments
export_args = (input_ids, attention_mask)

# 4. Dynamic sequence length, fixed batch size of 1
sequence_length = torch.export.Dim(
    "sequence_length",
    min=2,
    max=1024,
)

dynamic_shapes = {
    "input_ids": {1: sequence_length},
    "attention_mask": {1: sequence_length},
}

# 5. Export to ONNX
print("Exporting ModernBERT to ONNX...")

try:
    torch.onnx.export(
        model,
        args=export_args,
        f="modernbert_classification.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        external_data=True,
        report=True,
    )

    print("ONNX export completed.")

except Exception as e:
    print("ONNX export failed:")
    print(type(e).__name__, str(e))
    raise