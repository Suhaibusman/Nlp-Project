import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

# Get the prompt from the user
prompt = input("Enter your prompt: ")

# Encode the prompt and convert it to a tensor
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text from the model
output = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1)

# Decode the output and print it
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)