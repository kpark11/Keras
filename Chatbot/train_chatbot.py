import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD,RMSprop,Adam
import random

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
'''
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
'''
train_x = np.zeros(shape=(len(training), len(training[0][0])))
train_y = np.zeros(shape=(len(training), len(training[0][1])))
for i in range(len(training[0][1])):
    for n in range(len(training[i][0])):
        train_x[i][n] = training[i][0][n]
    
for i in range(len(training[0][1])):
    for n in range(len(training[i][1])):
        train_y[i][n] = training[i][1][n]
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains 
# number of neurons equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128*2, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64*2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#sgd = SGD(learning_rate=0.001, decay=1e-8, momentum=0.5, nesterov=True)
#rmsprop = RMSprop(learning_rate=0.01,rho=0.01,momentum=0.1,epsilon=1e-07)
adam = Adam(learning_rate=0.05,beta_1=0.7,beta_2=0.999,epsilon=1e-07)

model.compile(loss='BinaryCrossentropy', optimizer=adam, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(train_x, train_y, epochs=400, batch_size=5*2, verbose=1)
model.compile(loss='BinaryCrossentropy', optimizer=adam, metrics=['accuracy'])

model.save('chatbot_model.h5', hist)

print("model created")
