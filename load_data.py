import os
from datasets import load_dataset
from huggingface_hub import login

token = os.getenv("HUGGINGFACE_TOKEN")
login(token=token)

ds = load_dataset("openai/gsm8k", "main", split="train")
print(ds)

for sample in [sample for i, sample in enumerate(ds) if i < 5]:
    print("-----------")
    print("question:", sample["question"])
    print("answer:", sample["answer"])
