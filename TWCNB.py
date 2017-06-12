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

corpus_words = {}
class_words = {}
lemmatized_sentences = []

# Turn a list into a set of unique items and then a list again to remove duplications
classes = list(set([a['class'] for a in training_data]))

for c in classes:
    class_words[c] = []

# Loop through each sentence in our training data
for data in training_data:
    # Tokenize each sentence into words
    sentence = set()
    for word in nltk.word_tokenize(data['sentence']):
        # ignore some things
        if word not in ["?", "'s"]:
            stemmed_word = stemmer.stem(word.lower())
            # Have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
            # Add the word to our words in class list
            sentence.add(stemmed_word)
            # This is frequency so we need to change this part
            class_words[data['class']].extend([stemmed_word])
    lemmatized_sentences.append(sentence)

def convertAllFrequencies():
    """
    Loop through all the words and convert the frequencies
    """
    global corpus_words
    for key, value in corpus_words.iteritems():
        corpus_words[key] = transformTermFrequency(value)

def transformTermFrequency(freq):
    """
    Adjust the given term's frequency to produce a more empirical distribution
    Note: We use this after the regular frequencies of all words are figured out
    We just adjust them
    :return: the terms adjusted frequency
    """
    return math.log10(freq + 1)

def inverseDocumentFrequency():
    """
    Recalculate frequencies based on the term's number of occurrences in document
    :return: Recalculated frequencies
    """
    global corpus_words
    for key, val in corpus_words.iteritems():
        numerator = len(training_data)
        denominator = 0 # I need to find a way to avoid this issue
        for sentence in lemmatized_sentences:
            denominator += wordInDocument(key, sentence)
        corpus_words[key] = val * math.log10(numerator/denominator)

# def lengthTransformation():
#     """
#     Calculate the frequencies based on the terms frequency per document
#     And then recalculate the entire frequency.
#     :return: Recalculated frequencies
#     """
#     global corpus_words I"M SKIPPING STEP 3 BECAUSE MULTINOMIAL MODEL DOES IT VERY WELL
# AND CHANGES ARE SUBTLE


def wordInDocument(word, sentence):
    """
    Check if the word passed in is in the document.
    :param word: The word being checked if the document contains it
    :return: If word exists in document return 1 else 0
    """
    if word in sentence:
        return 1
    return 0

for key,value in corpus_words.iteritems():
    print "Key: ", key," ", "Value: ", value
convertAllFrequencies()

print "AFTER CONVERSION!"

for key,value in corpus_words.iteritems():
    print "Key: ", key," ", "Value: ", value
# print("Class words: {0}").format(class_words)

inverseDocumentFrequency()
print "After Inverse Doc Frequency"

for key,value in corpus_words.iteritems():
    print "Key: ", key," ", "Value: ", value

