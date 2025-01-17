import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import os 
from datasets import load_dataset
import pandas as pd
import json
import time

os.environ['HUGGING_FACE_HUB_TOKEN'] = os.getenv("HUGGING_FACE_HUB_TOKEN")
model_path = "DrissDo/Qwen2.5-3B-JohnMa"

model = AutoModelForCausalLM.from_pretrained("/home/ltnga/ITDSIU21079/src/llama_v2/checkpoint-100")

tokenizer_path = "DrissDo/Qwen2.5-3B-JohnMa"

tokenizer = AutoTokenizer.from_pretrained("/home/ltnga/ITDSIU21079/src/llama_v2/checkpoint-100")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

ids = []
questions = []
choices_A = []
choices_B = []
choices_C = []
choices_D = []
choices_E = []
answers=[]

with open('/home/ltnga/ITDSIU21079/VMLU/vmlu_v1.5/valid.jsonl', "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        ids.append(data["id"])
        questions.append(data["question"])
        answers.append(data["answer"])
        choices = data["choices"]
        try:
            choices_A.append(choices[0])
        except:
            choices_A.append('')
        try:
            choices_B.append(choices[1])
        except:
            choices_B.append('')
        try:
            choices_C.append(choices[2])
        except:
            choices_C.append('')
        try:
            choices_D.append(choices[3])
        except:
            choices_D.append('')
        try:
            choices_E.append(choices[4])
        except:
            choices_E.append('')

df = pd.DataFrame({
    "id": ids,
    "prompt": questions,
    "A": choices_A,
    "B": choices_B,
    "C": choices_C,
    "D": choices_D,
    "anwser": answers
})

df["anwser"].value_counts()

from string import Template

preamble = 'Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D) của câu hỏi trắc nghiệm sau: '

template = Template('$preamble\n\n$prompt\n\n $a\n $b\n $c\n $d \nĐáp án:')

def format_input(df, idx):
    prompt = df.loc[idx, 'prompt']
    a = df.loc[idx, 'A']
    b = df.loc[idx, 'B']
    c = df.loc[idx, 'C']
    d = df.loc[idx, 'D']

    input_text = template.substitute(
        preamble=preamble, prompt=prompt, a=a, b=b, c=c, d=d)

    return input_text

from tqdm import tqdm

start = time.time()

for idx in df.index:
    inputs = tokenizer(format_input(df, idx), return_tensors="pt", return_token_type_ids=False).to(device)
    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1)

    answer_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    last_element = answer_decoded[-1]
    answer = last_element.split()[-1]
    answers.append(answer)

end = time.time()
duration = end - start
print('Time taken for running inference: ', duration)

df["answer_bot"] = answers

df['is_correct'] = df['anwser'] == df['answer_bot']
correct_count = df['is_correct'].sum()
total_count = len(df)
accuracy = (correct_count / total_count) * 100
print(f"Phần trăm đúng: {accuracy:.2f}%")
