from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import random
from datetime import datetime
from nltk.tokenize import word_tokenize
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This allows all origins to make requests to your Flask app

# Download necessary NLTK data
nltk.download('punkt')

# Sample data and model setup
qa_pairs = [
    # General greetings
    {"question": "hello", "answer": "Hello! How can I help you today?"},
    {"question": "hi there", "answer": "Hi! How are you doing?"},
    {"question": "hey", "answer": "Hey there! What can I do for you?"},
    {"question": "good morning", "answer": "Good morning! How can I assist you today?"},
    {"question": "good afternoon", "answer": "Good afternoon! What can I help you with?"},
    {"question": "good evening", "answer": "Good evening! How may I assist you?"},
    
    # Status inquiries
    {"question": "how are you", "answer": "I'm doing well, thank you for asking! How about you?"},
    {"question": "i am fine", "answer": "That's great to hear! Is there anything I can help you with?"},
    {"question": "i am good", "answer": "Glad to hear that! What can I do for you today?"},
    {"question": "how are you doing", "answer": "I'm functioning perfectly! How can I assist you?"},
    
    # Identity questions
    {"question": "what is your name", "answer": "I'm a simple chatbot designed to help answer your questions."},
    {"question": "who are you", "answer": "I'm an AI assistant created to chat and provide information."},
    {"question": "what can you do", "answer": "I can answer questions, chat with you, perform calculations, tell jokes, and more!"},
    
    # Farewells
    {"question": "goodbye", "answer": "Goodbye! Have a nice day!"},
    {"question": "bye", "answer": "Bye! Feel free to chat again if you need anything."},
    {"question": "see you later", "answer": "See you later! Come back anytime."},
    
    # Gratitude
    {"question": "thank you", "answer": "You're welcome! Is there anything else you'd like to know?"},
    {"question": "thanks", "answer": "You're welcome! Happy to help."},
    {"question": "appreciate it", "answer": "Glad I could help! Let me know if you need anything else."},
    
    # Help
    {"question": "help", "answer": "I can answer questions, chat, perform calculations, tell jokes, give weather info, and more. Just ask!"},
    {"question": "what can i ask", "answer": "You can ask me general questions, math problems, about the weather, time, or even for a joke!"},
    
    # Additional knowledge
    {"question": "what is python", "answer": "Python is a popular programming language known for its simplicity and readability. It's widely used in data science, web development, and automation."},
    {"question": "what is machine learning", "answer": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed."},
    {"question": "what is chatgpt", "answer": "ChatGPT is an AI language model developed by OpenAI that can generate human-like text based on the prompts it receives."},
    {"question": "what is tensorflow", "answer": "TensorFlow is an open-source machine learning framework developed by Google that's commonly used for training and deploying neural networks."},
    
    # Jokes request
    {"question": "tell me a joke", "answer": "JOKE_PLACEHOLDER"},
    {"question": "i want to hear a joke", "answer": "JOKE_PLACEHOLDER"},
    {"question": "do you know any jokes", "answer": "JOKE_PLACEHOLDER"},
    
    # Weather request
    {"question": "what is the weather", "answer": "WEATHER_PLACEHOLDER"},
    {"question": "how is the weather today", "answer": "WEATHER_PLACEHOLDER"},
    {"question": "is it going to rain", "answer": "WEATHER_PLACEHOLDER"},
    
    # Time request
    {"question": "what time is it", "answer": "TIME_PLACEHOLDER"},
    {"question": "tell me the time", "answer": "TIME_PLACEHOLDER"},
    {"question": "what is the date today", "answer": "DATE_PLACEHOLDER"},
    
    # Math related
    {"question": "can you do math", "answer": "Yes, I can perform basic math operations like addition, subtraction, multiplication, and division. Just type an expression like '5 + 3' or 'what is 10 divided by 2'."},
    {"question": "solve math problem", "answer": "Sure, I can solve basic math problems. Just type an expression like '5 + 3' or 'what is 10 divided by 2'."}
]

# Extract questions and answers
questions = [pair["question"] for pair in qa_pairs]
answers = [pair["answer"] for pair in qa_pairs]

# Preprocess the data
def preprocess(text):
    return ' '.join([word.lower() for word in word_tokenize(text) if word.isalnum()])

# Preprocess the questions before fitting the model
processed_questions = [preprocess(question) for question in questions]

# Create the TF-IDF vectorizer and transform the processed questions
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(processed_questions)

# Check if the input is a math expression
def is_math_expression(text):
    # Simple regex to detect basic math expressions
    return bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', text)) or bool(re.search(r'what is \d+\s*[\+\-\*\/]\s*\d+', text.lower()))

# Evaluate simple math expressions
def evaluate_math(text):
    try:
        # Extract the math expression using regex
        match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', text)
        if match:
            num1 = int(match.group(1))
            operator = match.group(2)
            num2 = int(match.group(3))
            
            if operator == '+':
                return f"{num1} + {num2} = {num1 + num2}"
            elif operator == '-':
                return f"{num1} - {num2} = {num1 - num2}"
            elif operator == '*':
                return f"{num1} * {num2} = {num1 * num2}"
            elif operator == '/':
                if num2 == 0:
                    return "Sorry, I can't divide by zero."
                return f"{num1} / {num2} = {num1 / num2}"
    except:
        pass
    
    # Try to extract from natural language expression
    match = re.search(r'what is (\d+)\s*(plus|minus|times|divided by)\s*(\d+)', text.lower())
    if match:
        try:
            num1 = int(match.group(1))
            operation = match.group(2)
            num2 = int(match.group(3))
            
            if operation == 'plus':
                return f"{num1} + {num2} = {num1 + num2}"
            elif operation == 'minus':
                return f"{num1} - {num2} = {num1 - num2}"
            elif operation == 'times':
                return f"{num1} * {num2} = {num1 * num2}"
            elif operation == 'divided by':
                if num2 == 0:
                    return "Sorry, I can't divide by zero."
                return f"{num1} / {num2} = {num1 / num2}"
        except:
            pass
    
    return "I'm sorry, I couldn't understand that math expression. I can handle basic operations like addition, subtraction, multiplication, and division."

# Generate a random joke
def get_joke():
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!",
        "Why was the math book sad? It had too many problems.",
        "What do you call a parade of rabbits hopping backwards? A receding hare-line.",
        "Why don't skeletons fight each other? They don't have the guts.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
        "How do you organize a space party? You planet.",
        "I'm reading a book about anti-gravity. It's impossible to put down!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!"
    ]
    return random.choice(jokes)

# Simulate weather information
def get_weather():
    conditions = ["sunny", "cloudy", "rainy", "snowy", "windy", "partly cloudy", "clear"]
    temperatures = range(10, 32)
    
    condition = random.choice(conditions)
    temperature = random.choice(temperatures)
    
    return f"It's currently {condition} with a temperature of {temperature}°C. (Note: This is simulated data since I don't have access to real-time weather information.)"

# Get current time and date
def get_time():
    current_time = datetime.now().strftime("%H:%M:%S")
    return f"The current time is {current_time}."

def get_date():
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    return f"Today's date is {current_date}."

# Check for specific intents
def check_special_intent(user_input):
    user_input_lower = user_input.lower()
    
    # Check for joke request
    if "joke" in user_input_lower or "funny" in user_input_lower:
        return get_joke()
    
    # Check for weather request
    if "weather" in user_input_lower or "temperature" in user_input_lower or "raining" in user_input_lower:
        return get_weather()
    
    # Check for time request
    if "time" in user_input_lower and "what" in user_input_lower:
        return get_time()
    
    # Check for date request
    if "date" in user_input_lower and "today" in user_input_lower:
        return get_date()
    
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if request.method != 'POST':
        return jsonify({"error": "Method Not Allowed, only POST allowed."}), 405
    
    # Get user input from the request
    user_input = request.json.get('message', '')
    
    if not user_input:
        return jsonify({"error": "No message provided."}), 400
    
    # Check for special intents first
    special_response = check_special_intent(user_input)
    if special_response:
        return jsonify({"response": special_response})
    
    # Check if it's a math question
    if is_math_expression(user_input):
        response = evaluate_math(user_input)
        return jsonify({"response": response})
    
    # Preprocess user input
    user_input_processed = preprocess(user_input)
    user_input_vector = vectorizer.transform([user_input_processed])
    
    # Find the most similar response using cosine similarity
    cosine_similarities = cosine_similarity(user_input_vector, vectors)
    best_match_index = cosine_similarities.argmax()
    
    # Set a minimum similarity threshold
    best_similarity = cosine_similarities[0][best_match_index]
    
    if best_similarity > 0.15:  # This threshold can be adjusted
        response = answers[best_match_index]
        
        # Handle placeholders
        if response == "JOKE_PLACEHOLDER":
            response = get_joke()
        elif response == "WEATHER_PLACEHOLDER":
            response = get_weather()
        elif response == "TIME_PLACEHOLDER":
            response = get_time()
        elif response == "DATE_PLACEHOLDER":
            response = get_date()
    else:
        response = "I'm not sure I understand. Could you rephrase that or ask me something else?"
    
    # Return the best matching response
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)