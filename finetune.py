from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Loading pre-trained GPT-2 model and tokenizer
model_path = "C:/Users/irosh/fine-tuned-gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Loading and tokenizing dataset
dataset_path = "C:\\Users\\irosh\\OneDrive\\Documents\\chatapp\\dataset3.txt"
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=dataset_path,
    block_size=128  
)

# directory where the fine-tuned model will be saved
output_dir = "C:/Users/irosh/fine-tuned-gpt2-large"

# training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,  
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset,
)

# Fine-tune model
trainer.train()

# Save fine-tuned model 
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
