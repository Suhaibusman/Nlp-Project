import torch
from transformers import AutoTokenizer, GPT2DoubleHeadsModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2DoubleHeadsModel.from_pretrained("gpt2")

num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})

prompt = input("Enter your prompt: ")

input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
