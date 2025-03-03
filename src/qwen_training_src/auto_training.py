from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B", 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for training
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen-vietnam-traffic",
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()