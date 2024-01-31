from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlx_lm import load, generate

app = FastAPI()

# Load the model when the server starts
tokenizer_config = {"trust_remote_code": True}
model, tokenizer = load("mlx-community/stablelm-2-zephyr-1_6b-4bit", tokenizer_config=tokenizer_config)

@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 500)

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
    # Assuming generate function does not inherently support streaming
    response_text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True)

    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            }
        }]
    }
    return JSONResponse(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
