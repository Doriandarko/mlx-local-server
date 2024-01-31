from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
]

while True:
    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=history,
        max_tokens = 500
    )

    # Accessing the message content as an object property
    response_content = completion.choices[0].message.content
    print(response_content)

    # Append the assistant's response to the history
    history.append({"role": "assistant", "content": response_content})

    # Get the next user input and add it to the history
    user_input = input("You: ")
    history.append({"role": "user", "content": user_input})


