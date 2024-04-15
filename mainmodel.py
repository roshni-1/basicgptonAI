from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Paths for the model and dataset
model_name = "gpt2-large"
model_path = f"./{model_name}"
dataset_path = "C:/Users/irosh/OneDrive/Documents/traindataset2.txt"

# Downloading the GPT-2 large model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Loading the dataset
dataset = TextDataset(tokenizer=tokenizer, file_path=dataset_path, block_size=128)

# Initializing the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Initializing Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()
