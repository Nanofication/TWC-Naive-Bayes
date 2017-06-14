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

import re
import math
import nltk
from nltk.stem.lancaster import LancasterStemmer

from ClassifierClass import Class
from DataClass import TrainingData
from DocumentClass import Document

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

CLASS_WORDS = {}

CLASSES = list(set([a['class'] for a in training_data]))

CLASS_DICT = {}

LARGE_NEGATIVE_NUMBER = -2147483648 # You cannot have log 0 in math or code

TRAINING_DATA_STATS = TrainingData("Training Data 1")

def initializeData():
    global CLASS_WORDS
    global CLASSES
    global CLASS_DICT
    global TRAINING_DATA_STATS

    for i in range(len(CLASSES)):
        CLASS_DICT[CLASSES[i]] = Class(CLASSES[i])

    for data in training_data:
        # Tokenize each sentence into words
        sentence = removeSpecialCharacters(data['sentence'])
        sentence = nltk.word_tokenize(sentence)
        sentence = lemmatizeSentence(sentence)
        doc_num = CLASS_DICT[data['class']].getTotalDocuments()

        document = Document(doc_num, data['class'], sentence)
        CLASS_DICT[data['class']].addDocument(document)
        TRAINING_DATA_STATS.training_docs.append(sentence)

        for word in sentence:
            # Have we not seen this word already?
            CLASS_DICT[data['class']].addToTotalClassWordFreq(word)
            # Add the word to our words in class list
            CLASS_DICT[data['class']].documents[doc_num].addToDocsWordFreq(word)

            # This is frequency so we need to change this part
            CLASS_DICT[data['class']].addWords([word])

            TRAINING_DATA_STATS.addToTotalWordFreq(word)
            TRAINING_DATA_STATS.addWords([word])

    # for key, val in CLASS_DICT.iteritems():
    #     print key, ": ", val.class_name, " ", val.words, " | ", val.word_freq, " | ", val.word_count
    #
    # for key, val in TRAINING_DATA_STATS.word_freq.iteritems():
    #     print key, ": ", val

def lemmatizeSentence(sentence):
    """
    Reduce each word in the sentence to its base case
    :param sentence: The entire sentence to be lemmatized
    :return: The sentence with words in its base case
    """
    for i in range(len(sentence)):
        sentence[i] = stemmer.stem(sentence[i])
    return sentence

def removeSpecialCharacters(sentence):
    pattern = re.compile('[^A-Za-z0-9 ]+')
    sentence = re.sub(pattern, '', sentence)
    return sentence

def wordInDocument(word, sentence):
    """
    Check if the word passed in is in the document.
    :param word: The word being checked if the document contains it
    :return: If word exists in document return 1 else 0
    """
    if word in sentence:
        return 1
    return 0

def checkAllInstancesOfWord(word):
    """
    Check the number of times the word occurs in the list of training documents
    :param word: Word to be checked in each document
    :return: Number of docs with one or more instance of word
    """
    total = 0
    for doc in TRAINING_DATA_STATS.training_docs:
        total += wordInDocument(word, doc)
    return total

###### Transformed Weighted Complement Naive Bayes #####

def transformTermFrequency():
    """
    Transform all word weights from each class's individual documents
    using the equation d_ij = log(d_ij + 1)

    Return class with updated weights
    """
    global CLASS_DICT
    for key, val in CLASS_DICT.iteritems():
        for doc in val.documents:
            doc.normalizeWordFreq()
            print doc.normalized_word_freq
            print doc.word_freq

def transformByDocFrequency():
    """
    Transform the word weight based on frequency of the word occuring
    in all documents.

    Caution: Runtime O(n^3)

    Follow this equation d_ij = d_ij * log((Sum of all docs)/ Sum of if word occurs in doc)
    :return: The class with updated weights.
    """
    global CLASS_DICT
    global TRAINING_DATA_STATS

    numerator = len(TRAINING_DATA_STATS.training_docs)

    for key, val in CLASS_DICT.iteritems():
        for doc in val.documents:
            for k, v in doc.normalized_word_freq.iteritems():
                denominator = checkAllInstancesOfWord(k)
                doc.normalized_word_freq[k] = v * math.log10(numerator/denominator)
#
# def transformByLength():
#     """
#     Convert d_ij by dividing this value with the square root of
#     The sum of the square of number of times a word exists in the document
#     :return:
#     """
#     global CLASS_DICT
#     global TRAINING_DATA_STATS
#
#     for key, val in CLASS_DICT.iteritems():
#         for doc in val.documents:
#             for k, v in doc.normalized_word_freq.iteritems():
#                 denominator = math.sqrt((TRAINING_DATA_STATS.word_freq[k])**2)
#                 doc.normalized_word_freq[k] = val / denominator
#                 print doc.normalized_word_freq
#
#   Multinomial Naive Bayes solves this issue, but look into it in future

def skewDataBiasHandler():
    """
    Set the parameter weights of all words into a less biased value

    Use the equation:
    Normalized frequency = (SUM of specific word's frequency in all training sets - word frequency in current class)
    Divided by (SUM of all word's frequency - words frequency in the given class)
    """

    global TRAINING_DATA_STATS
    global CLASS_DICT
    global CLASSES

    alpha = 1

    # for c in CLASSES:
    #     for doc in CLASS_DICT[c].documents:
    #         print doc.classification, ": ", doc.data

    for key, val in TRAINING_DATA_STATS.word_freq.iteritems():
        numerator = 0
        denominator = 0
        totalAlpha = 0
        for c in CLASSES:
            if key != c:
                for doc in CLASS_DICT[c].documents:
                    # I have each individual document
                    try:
                        numerator += doc.normalized_word_freq[key] + alpha
                        totalAlpha += 1
                    except:
                        pass
                    denominator += doc.sumNormalizedFreq() + totalAlpha
        # print key, "| Numerator: ", numerator, "Denominator: ", denominator
        TRAINING_DATA_STATS.word_normalized_freq[key] = numerator/denominator

    # for key, val in TRAINING_DATA_STATS.word_normalized_freq.iteritems():
    #     print key, "||| ", val

def setWordWeights():
    """
    Pass all the weights after normalized and do a conversion.
    :return: Store new value into training data's word weight
    """
    global TRAINING_DATA_STATS

    for key, val in TRAINING_DATA_STATS.word_normalized_freq.iteritems():
        if val != 0:
            TRAINING_DATA_STATS.word_weight[key] = math.log10(val)
        else:
            TRAINING_DATA_STATS.word_weight[key] = LARGE_NEGATIVE_NUMBER

def normalizeWordWeights():
    """
    Normalize the word weights stored from training data
    :return: The updated values of word weights
    """
    global TRAINING_DATA_STATS
    total = TRAINING_DATA_STATS.getTotalWordWeight()

    for key, val in TRAINING_DATA_STATS.word_weight.iteritems():
        TRAINING_DATA_STATS.word_weight[key] = val / total

    for key, val in TRAINING_DATA_STATS.word_weight.iteritems():
        print key, ": ", val

####### MULTINOMIAL NAIVE BAYES TEST CODE #######

def calculate_class_score(sentence, class_name, show_details=True):
    global CLASS_DICT
    score = 0

    for word in nltk.word_tokenize(sentence):
        if stemmer.stem(word.lower()) in CLASS_DICT[class_name].word_freq:
            # Treat each word with relative weight
            current_score = 1.0 / TRAINING_DATA_STATS.word_freq[stemmer.stem(word.lower())]
            score += current_score

            if show_details:
                print (
                "   match: %s (%s)" % (stemmer.stem(word.lower()), 1.0 / TRAINING_DATA_STATS.word_freq[stemmer.stem(word.lower())]))
    return score

def classify(sentence):
    global CLASS_DICT

    high_class = None
    high_score = 0
    # loop through our classes
    for c in CLASS_DICT.keys():
        # calculate score of sentence for each class
        score = calculate_class_score(sentence, c)
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score

    return high_class, high_score
#####################################################

# def classifyTWCNB(sentence):
#

if __name__ == "__main__":
    initializeData()
    transformTermFrequency()
    transformByDocFrequency()
    #transformByLength()
    skewDataBiasHandler()
    setWordWeights()
    normalizeWordWeights()

    # Test using multinomial naive bayes
    # sentence = "Hello how are you doing?"
    #
    # print classify(sentence)
