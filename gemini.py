from flask import Flask, render_template, request, jsonify, session
import requests
import os
from dotenv import load_dotenv  # Import dotenv to load environment variables

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_default_secret_key")  # Use env variable or default

# Get API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found in environment variables.")

# Construct API URL
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# Define headers
headers = {
    "Content-Type": "application/json"
}

prompt_count = 0

# Static chat history (for demonstration purposes)
static_chat_history = [
    {"role": "user", "content": "Hello, I have been feeling very tired lately."},
    {"role": "assistant", "content": "I'm sorry to hear that. Can you tell me more about your symptoms? How long have you been feeling this way?"},
    {"role": "user", "content": "It's been about two weeks. I also have trouble sleeping."},
    {"role": "assistant", "content": "Thank you for sharing. Have you noticed any other symptoms, such as headaches or changes in appetite?"}
]

def generate_response(prompt_text):
    global prompt_count
    prompt_count += 1

    # Add the new user message to the static chat history
    static_chat_history.append({"role": "user", "content": prompt_text})

    # Prepare the conversation history for the API request
    conversation_history = [
        {"role": "user" if msg["role"] == "user" else "assistant", "parts": [{"text": msg["content"]}]}
        for msg in static_chat_history
    ]

    # Define the request payload for Gemini API
    data = {
        "contents": conversation_history
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        bot_response = response_data['candidates'][0]['content']['parts'][0]['text']

        # Add the bot's response to the static chat history
        static_chat_history.append({"role": "assistant", "content": bot_response})

        return bot_response
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
