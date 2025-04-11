from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the model with specific parameters
model_name = "/home/ltnga/ITDSIU21079/src/qwen_training_src/qwen-vietnam-traffic-merged"
llm = LLM(
    model=model_name,
    trust_remote_code=True,  # Important for Qwen models
    dtype="bfloat16",        # Match your training dtype
    tensor_parallel_size=1   # Adjust based on GPU count
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

@app.get("/")
def read_root():
    return {"message": "Vietnam Traffic QA API is running. Use POST /generate to interact with the model."}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    try:
        # Format the prompt to match your training format
        formatted_prompt = f"Hỏi: {request.prompt}\n\n"
        
        # Update sampling parameters
        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        outputs = llm.generate(formatted_prompt, params)
        return {"generated_text": outputs[0].text}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)