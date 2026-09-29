from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Step 3
# Generate data 
from datasets import Dataset, load_dataset

# Create a simple dataset
data = {
    'inputs': [
        "summarize: This is a sentence.",
        "summarize: This is another sentence.",
        "summarize: This is a test sentence."
    ],
    'targets': [
        "This sentence has a summary.",
        "This is another sentence with a summary.",
        "This is a test sentence with a summary."
    ]
}

# Convert the data to a Dataset
dataset = Dataset.from_dict(data)

# Step 4
# Fine-Tuning the Model

from transformers import TrainingArguments, Trainer

# Configure training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
)



# Step 5
# Train the model
trainer.train()


# Step 6
# Evaluate the model
evaluation_results = trainer.evaluate()

# Print the evaluation results
for key, value in evaluation_results.items():
    print(f"{key}: {value}")



# Step 7
# Generate text
generated_text = trainer.predict(["summarize: This is a sentence."])

# Print the generated text
print(generated_text.predictions[0])
