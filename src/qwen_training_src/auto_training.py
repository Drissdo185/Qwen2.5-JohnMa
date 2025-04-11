import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from typing import Dict, List, Union

# Function to load and prepare the dataset
def prepare_dataset(json_file: str) -> Dataset:
    """
    Load and prepare the dataset from a JSON file.
    
    Args:
        json_file: Path to the JSON file containing the dataset
        
    Returns:
        A Hugging Face Dataset
    """
    # Load the data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    # Process each example
    for item in data:
        # Format: Question + reasoning + final answer
        prompt = f"Hỏi: {item['question']}\n\n"
        response = f"Suy luận ban đầu: {item['initial_reasoning']}\n\n"
        
        # Add iterations
        for i, iteration in enumerate(item['iterations']):
            response += f"Bước {i+1}:\n"
            response += f"Truy vấn: {iteration['query']}\n"
            response += "Tài liệu tham khảo:\n"
            for doc in iteration['retrieved_documents']:
                response += f"- {doc}\n"
            response += f"Suy luận: {iteration['reasoning']}\n\n"
        
        # Add final answer
        response += f"Câu trả lời cuối cùng: {item['final_answer']}"
        
        # Create a formatted example
        processed_data.append({
            "prompt": prompt,
            "response": response,
            "combined": prompt + response
        })
    
    # Convert to HF Dataset
    return Dataset.from_list(processed_data)

# Function to tokenize the dataset
def tokenize_dataset(examples: Dict[str, List], tokenizer, max_length: int = 512) -> Dict[str, List]:
    """
    Tokenize the dataset examples.
    
    Args:
        examples: Dictionary containing the examples
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized examples
    """
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["combined"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create the labels (same as input_ids for causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Make sure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare the dataset
train_data = prepare_dataset("/home/ltnga/ITDSIU21079/auto_training.json")

# Split into train and eval
train_test_split = train_data.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Tokenize datasets
max_length = 768  # Adjust based on your requirements
tokenized_train = train_dataset.map(
    lambda examples: tokenize_dataset(examples, tokenizer, max_length),
    batched=True,
    remove_columns=train_dataset.column_names
)
tokenized_eval = eval_dataset.map(
    lambda examples: tokenize_dataset(examples, tokenizer, max_length),
    batched=True,
    remove_columns=eval_dataset.column_names
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Include more projection matrices
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're using a causal language model
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen-vietnam-traffic",
    learning_rate=2e-3,
    max_steps=100,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10,
    fp16=True,
    load_best_model_at_end=True,
    report_to="tensorboard",
    save_total_limit=3,
    warmup_steps=100,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./qwen-vietnam-traffic-final")

# Save LoRA adapter separately for easier loading
model.save_pretrained("./qwen-vietnam-traffic-adapter")

# ----------------------------------------
# MERGING THE MODEL - ADD THIS SECTION
# ----------------------------------------
print("Starting the model merging process...")

# Get the original dtype of the model
original_dtype = model.dtype

# Unload the current model to free up GPU memory
del model
torch.cuda.empty_cache()

# Reload the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use the same dtype as during training
    device_map="auto"
)

# Load the trained adapter
adapter_path = "./qwen-vietnam-traffic-adapter"
print(f"Loading adapter from {adapter_path}")
adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge weights
print("Merging adapter weights with base model...")
merged_model = adapter_model.merge_and_unload()

# Convert back to original dtype if necessary
if merged_model.dtype != original_dtype:
    merged_model = merged_model.to(dtype=original_dtype)

# Save the fully merged model
merged_model_path = "./qwen-vietnam-traffic-merged"
print(f"Saving merged model to {merged_model_path}")
merged_model.save_pretrained(merged_model_path)

# Save tokenizer with the merged model
tokenizer.save_pretrained(merged_model_path)

print(f"Model successfully merged and saved to {merged_model_path}")