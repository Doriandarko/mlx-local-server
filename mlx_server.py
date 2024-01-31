from flask import Flask, request, jsonify
from mlx_lm import load, generate

app = Flask(__name__)

# Load the model when the server starts
tokenizer_config = {"trust_remote_code": True}
model, tokenizer = load("mlx-community/stablelm-2-zephyr-1_6b-4bit", tokenizer_config=tokenizer_config)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json()

    # Extract messages from the JSON request and construct the prompt
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 500)

    prompt = ""
    for message in messages:
        role = message.get('role')
        content = message.get('content')
        prompt += f"{role}: {content}\n"

    # Check if a prompt was constructed
    if prompt:
        response_text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        # Format the response to mimic OpenAI's API structure
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                }
            }]
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'No messages provided'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1234)
