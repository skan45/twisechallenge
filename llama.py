import requests

def talk_to_ollama(prompt):
    url = "http://127.0.0.1:11434"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "your_model_name",  # Replace with the model name you want to use (e.g., 'llama' or custom).
        "prompt": prompt,
        "stream": False,  # Set to True if you want a streamed response.
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        return data.get("response", "No response from Ollama.")
    except requests.RequestException as e:
        return f"An error occurred: {e}"

# Example usage
user_prompt = "Hello, how are you?"
response = talk_to_ollama(user_prompt)
print(f"Ollama: {response}")
