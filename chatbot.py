import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import LancasterStemmer
import pickle
import random

stemmer = LancasterStemmer()

model = tf.keras.models.load_model('chatbot_model.h5')
with open('intents.json') as f:
    data = json.load(f)
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

def bag_of_words(user_input):
    tokens = nltk.word_tokenize(user_input.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    bag = [1 if w in stemmed else 0 for w in words]
    return np.array(bag)

def predict_intent(user_input):
    bow = bag_of_words(user_input)
    result = model.predict(np.array([bow]))[0]
    threshold = 0.7
    results = [[i, r] for i, r in enumerate(result) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return labels[results[0][0]] if results else "support"

def get_response(intent_tag):
    for intent in data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that. Can you rephrase?"

print("Chatbot is running! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    intent = predict_intent(user_input)
    response = get_response(intent)
    print(f"Bot: {response}")