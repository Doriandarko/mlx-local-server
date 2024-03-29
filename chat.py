from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

completion = client.chat.completions.create(
    model="mlx-community/stablelm-2-zephyr-1_6b-4bit",  # Use 'model' to specify the model identifier
    messages=[
        {"role": "system", "content": "Always answer in rhymes."},
        {"role": "user", "content": "Introduce yourself."}
    ],
    max_tokens=500
)

# Accessing the message content as an object property
print(completion.choices[0].message.content)
