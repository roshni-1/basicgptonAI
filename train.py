#importing libraries
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Defining a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, tokenizer, text_data, max_length):
        self.tokenizer = tokenizer
        self.text_data = text_data
        self.max_length = max_length

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        # Tokenize the text sample
        inputs = self.tokenizer.encode_plus(
            self.text_data[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs

# Loading and preprocessing the data
text_data = []
with open("C:\\Users\\irosh\\OneDrive\\Documents\\testtext.txt", 'r', encoding='utf-8') as file:
    text_data.append(file.read())

max_length = 512  # or any other desired maximum length

# Defining the GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token  # or any other token you prefer
tokenizer.model_max_length = max_length  # Set the maximum length for tokenization

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Preparing the dataset
dataset = CustomDataset(tokenizer, text_data, max_length)

# Defining the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Defining training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initializing Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Training the model
trainer.train()

# Saving the trained model
model.save_pretrained("./gpt2-trained")

# Generating text using the trained model
prompt = "your prompt"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
output = model.generate(input_ids, max_length=50, temperature=1.0)

# Decoding and print generated text
for i, sample_output in enumerate(output):
    print(f"Generated Text {i+1}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")
