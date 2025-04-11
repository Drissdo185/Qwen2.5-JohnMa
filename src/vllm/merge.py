import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")

print("Loading adapter...")
adapter_path = "./qwen-vietnam-traffic-adapter"
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging models...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained("./qwen-vietnam-traffic-merged")
tokenizer.save_pretrained("./qwen-vietnam-traffic-merged")

print("Model successfully merged and saved!")