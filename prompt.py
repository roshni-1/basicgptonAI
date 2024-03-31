from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Loading fine-tuned GPT-2 model
model = GPT2LMHeadModel.from_pretrained("./gpt2-trained")

# Loading GPT-2 tokenizer using original pretrained model name 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Providing prompt
prompt = "what is Artificial Intelligence"

# Tokenizing the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generating text based on prompt
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=1.0)

# Decoding and printing generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
