import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import nltk
from nltk.stem import LancasterStemmer
import pickle

nltk.download('punkt')
stemmer = LancasterStemmer()

with open('intents.json') as f:
    data = json.load(f)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = sorted(set([stemmer.stem(w.lower()) for w in words]))
labels = sorted(labels)

training = []
output = []
out_empty = [0] * len(labels)

for i, doc in enumerate(docs_x):
    bag = [1 if stemmer.stem(w.lower()) in [stemmer.stem(wd.lower()) for wd in doc] else 0 for w in words]
    out_row = out_empty[:]
    out_row[labels.index(docs_y[i])] = 1
    training.append(bag)
    output.append(out_row)

training = np.array(training)
output = np.array(output)

model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(len(training[0]),), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training, output, epochs=200, batch_size=8, verbose=1)
model.save('chatbot_model.h5')

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print("Model trained and saved!")