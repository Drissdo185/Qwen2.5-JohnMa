# Model Card for Qwen2.5-3B - John Ma

## Model Details
This model draws inspiration from John Ma, a lawyer in the TVB series Come Home Love, which I watched during my childhood. In the series, the filmmakers often included legal instructions at the end of each episode, providing valuable legal insights to viewers in Hong Kong. I found this approach both impactful and educational, sparking my motivation to create a similar resource.

This model is the result of my undergraduate thesis, designed to provide legal question-and-answer support tailored to Vietnam. It aims to enhance public understanding of legal matters, much like the series inspired greater legal awareness in its audience.

### Model Description


This model is based on the **Qwen/Qwen2.5-3B** architecture, fine-tuned using **Low-Rank Adaptation (LoRA)** for a causal language modeling task. 

The primary purpose of this model is to support legal question-and-answering tasks specific to Vietnam. It has been trained with the **VTSNLP/instruct_general_dataset** to improve its Vietnamese language capabilities, alongside a custom legal instruction dataset to enhance its understanding and response accuracy for Vietnam's legal domain. Additionally, the model is optimized with 4-bit quantization, allowing efficient deployment on cloud platforms or devices with limited hardware, without requiring a GPU.

- **Developed by:** [Do Thanh Dat - IU - HCMVNU]
- **Finetuned from model:** Qwen/Qwen2.5-3B
- **Language(s) (NLP):** Vietnamese
- **License:** [Specify license, e.g., Apache 2.0]

---

## Training Details

### Training Configuration

The LoRA configuration used during fine-tuning is as follows:

```python
config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)
```
### Training Procedure
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=4,
        num_train_epochs=3,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        save_steps=1000,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="qwen_v1",
        report_to="none",
    ),
)
```

### Hardware Type
NVIDIA A100 - 80GB

### Fine-Tune Method
Instruction Tuning
