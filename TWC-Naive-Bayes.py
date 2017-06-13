"""

Implementing Transformed Weighted Complement Naive Bayes for
classifying sentence.

Following research paper: Tackling the Poor Assumptions of Naive Bayes Text Classifiers

By:
Jason D. M. Rennie
Lawrence Shih
Jaime Teevan
David R. Karger

"""

import math

import nltk
from nltk.stem.lancaster import LancasterStemmer


# Word stemmer. Reduce words to the root forms for better classification
stemmer = LancasterStemmer()

# 3 classes of training data. Play around with this
training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"what's up?"})
training_data.append({"class":"greeting", "sentence":"hi"})
training_data.append({"class":"greeting", "sentence":"how are you doing?"})
training_data.append({"class":"greeting", "sentence":"what's new?"})
training_data.append({"class":"greeting", "sentence":"how's life?"})
training_data.append({"class":"greeting", "sentence":"how are you doing today?"})
training_data.append({"class":"greeting", "sentence":"good to see you"})
training_data.append({"class":"greeting", "sentence":"nice to see you"})
training_data.append({"class":"greeting", "sentence":"long time no see"})
training_data.append({"class":"greeting", "sentence":"it's been a while"})
training_data.append({"class":"greeting", "sentence":"nice to meet you"})
training_data.append({"class":"greeting", "sentence":"pleased to meet you"})
training_data.append({"class":"greeting", "sentence":"how do you do"})
training_data.append({"class":"greeting", "sentence":"yo"})
training_data.append({"class":"greeting", "sentence":"howdy"})
training_data.append({"class":"greeting", "sentence":"sup"})
# 20 training data


training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"peace"})
training_data.append({"class":"goodbye", "sentence":"catch you later"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"farewell"})
training_data.append({"class":"goodbye", "sentence":"have a good day"})
training_data.append({"class":"goodbye", "sentence":"take care"})
# 10 training datas
training_data.append({"class":"goodbye", "sentence":"bye!"})
training_data.append({"class":"goodbye", "sentence":"have a good one"})
training_data.append({"class":"goodbye", "sentence":"so long"})
training_data.append({"class":"goodbye", "sentence":"i'm out"})
training_data.append({"class":"goodbye", "sentence":"smell you later"})
training_data.append({"class":"goodbye", "sentence":"talk to you later"})
training_data.append({"class":"goodbye", "sentence":"take it easy"})
training_data.append({"class":"goodbye", "sentence":"i'm off"})
training_data.append({"class":"goodbye", "sentence":"until next time"})
training_data.append({"class":"goodbye", "sentence":"it was nice seeing you"})

training_data.append({"class":"goodbye", "sentence":"it's been real"})
training_data.append({"class":"goodbye", "sentence":"im out of here"})

training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})

training_data.append({"class":"email", "sentence":"what's your email address?"})
training_data.append({"class":"email", "sentence":"may I get your email?"})
training_data.append({"class":"email", "sentence":"can I have your email?"})
training_data.append({"class":"email", "sentence":"what's your email?"})
training_data.append({"class":"email", "sentence":"let me get your email"})
training_data.append({"class":"email", "sentence":"give me your email"})
training_data.append({"class":"email", "sentence":"i'll take your email address"})
training_data.append({"class":"email", "sentence":"can I have your business email?"})
training_data.append({"class":"email", "sentence":"your email address?"})
training_data.append({"class":"email", "sentence":"email please?"})
training_data.append({"class":"email", "sentence":"may I have your email?"})
training_data.append({"class":"email", "sentence":"can I get your email?"})