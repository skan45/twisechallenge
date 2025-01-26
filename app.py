from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)
from pipline1 import top_three
def ask_ollama(prompt):
    # Refined prompt to enhance the chatbot's responses
    a,b,x=top_three(prompt) 
    enhanced_prompt = f"User asks: '{prompt}', referring to similar questions: {a} and {b} for context. Please provide an answer in two sentences. For more information, visit: {x}"
    command = f'ollama run llama3.2 "{enhanced_prompt}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr.strip()}"
    return result.stdout.strip()
def fix_encoding(response):
    # This function fixes the encoding issues by decoding and re-encoding the string
    try:
        # Re-encode to bytes, and decode it properly to handle special characters
        return response.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, TypeError) as e:
        # If an error occurs, return the original string
        return response    

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = ask_ollama(user_input)
    res=fix_encoding(response)
    return jsonify({"response": res})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
