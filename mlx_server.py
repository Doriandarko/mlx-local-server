from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlx_lm import load, generate

app = FastAPI()

@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    data = await request.json()

    # Use 'model' instead of 'model_id' to get the model identifier from the request
    model_identifier = data.get('model')
    if not model_identifier:
        raise HTTPException(status_code=400, detail="Model identifier not provided")

    # Load the model dynamically based on the provided model identifier
    tokenizer_config = {"trust_remote_code": True}
    model, tokenizer = load(model_identifier, tokenizer_config=tokenizer_config)

    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 500)

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
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
