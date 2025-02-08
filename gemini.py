from flask import Flask, render_template, request, jsonify, session
import requests
import json
from pipeline import top_five

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

GEMINI_API_KEY = "AIzaSyAwRQZ2Rb9VfWmTWta8aqk_6YJX_1KdYrw"
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY

# Define the headers
headers = {
    "Content-Type": "application/json"
}

prompt_count = 0

def generate_response(prompt_text):
    global prompt_count
    prompt_count += 1  
    context_list, answers = top_five(prompt_text)
    system_instruction = (
        "You are a medical therapist assistant. Your job is to diagnose patients "
        "based on their symptoms and provide advice while reminding them to seek professional help.\n\n"
        "Patient's symptoms and concerns:\n"
    )

    # Integrate retrieved context and previous responses
    context_text = "\n".join(f"Context {i+1}: {context}" for i, context in enumerate(context_list[:2]))
    answer_text = "\n".join(f"Answer {i+1}: {answer}" for i, answer in enumerate(answers[:2]))

    # Add diagnostic response every second prompt
    if prompt_count % 2 == 0:
        diagnostic_instruction = "\n\nBased on the above information, provide a detailed diagnosis and recommendations."
    else:
        diagnostic_instruction = "\n\nAsk the patient for more details if needed."

    # Combine everything into a single prompt
    final_prompt = (
        f"{system_instruction}{prompt_text}\n\n"
        f"Relevant Medical Context:\n{context_text}\n\n"
        f"Previous Responses:\n{answer_text} "
        f"{diagnostic_instruction}"
    )

    # Define the request payload for Gemini API
    data = {
        "contents": [{"parts": [{"text": final_prompt}]}]
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        return response_data['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle chatbot responses
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Generate a response using the Gemini API
    bot_response = generate_response(user_input)

    # Store chat history in session
    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append({"user": user_input, "bot": bot_response})

    return jsonify({"response": bot_response, "chat_history": session["chat_history"]})

# Route to get full chat history
@app.route("/history", methods=["GET"])
def get_chat_history():
    return jsonify({"chat_history": session.get("chat_history", [])})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
