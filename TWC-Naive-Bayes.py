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
import random
import ReadData

import re
import math
import sys
import nltk
from nltk.stem.lancaster import LancasterStemmer

from ClassifierClass import Class
from DataClass import TrainingData
from DocumentClass import Document

# Word stemmer. Reduce words to the root forms for better classification
stemmer = LancasterStemmer()

# TESTING PURPOSES
ReadData.shuffleTrainingData()
training_data = ReadData.TRAINING_DATA[:800]
test_data = ReadData.TRAINING_DATA[800:]


# 3 classes of training data. Play around with this
# training_data = []
# training_data.append({"class":"greeting", "sentence":"how are you?"})
# training_data.append({"class":"greeting", "sentence":"how is your day?"})
# training_data.append({"class":"greeting", "sentence":"good day"})
# training_data.append({"class":"greeting", "sentence":"how is it going today?"})
# training_data.append({"class":"greeting", "sentence":"what's up?"})
# training_data.append({"class":"greeting", "sentence":"hi"})
# training_data.append({"class":"greeting", "sentence":"how are you doing?"})
# training_data.append({"class":"greeting", "sentence":"what's new?"})
# training_data.append({"class":"greeting", "sentence":"how's life?"})
# training_data.append({"class":"greeting", "sentence":"how are you doing today?"})
# training_data.append({"class":"greeting", "sentence":"good to see you"})
# training_data.append({"class":"greeting", "sentence":"nice to see you"})
# training_data.append({"class":"greeting", "sentence":"long time no see"})
# training_data.append({"class":"greeting", "sentence":"it's been a while"})
# training_data.append({"class":"greeting", "sentence":"nice to meet you"})
# training_data.append({"class":"greeting", "sentence":"pleased to meet you"})
# training_data.append({"class":"greeting", "sentence":"how do you do"})
# training_data.append({"class":"greeting", "sentence":"yo"})
# training_data.append({"class":"greeting", "sentence":"howdy"})
# training_data.append({"class":"greeting", "sentence":"sup"})
# # 20 training data
#
#
# training_data.append({"class":"goodbye", "sentence":"have a nice day"})
# training_data.append({"class":"goodbye", "sentence":"see you later"})
# training_data.append({"class":"goodbye", "sentence":"have a nice day"})
# training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
# training_data.append({"class":"goodbye", "sentence":"peace"})
# training_data.append({"class":"goodbye", "sentence":"catch you later"})
# training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
# training_data.append({"class":"goodbye", "sentence":"farewell"})
# training_data.append({"class":"goodbye", "sentence":"have a good day"})
# training_data.append({"class":"goodbye", "sentence":"take care"})
# # 10 training datas
# training_data.append({"class":"goodbye", "sentence":"bye!"})
# training_data.append({"class":"goodbye", "sentence":"have a good one"})
# training_data.append({"class":"goodbye", "sentence":"so long"})
# training_data.append({"class":"goodbye", "sentence":"i'm out"})
# training_data.append({"class":"goodbye", "sentence":"smell you later"})
# training_data.append({"class":"goodbye", "sentence":"talk to you later"})
# training_data.append({"class":"goodbye", "sentence":"take it easy"})
# training_data.append({"class":"goodbye", "sentence":"i'm off"})
# training_data.append({"class":"goodbye", "sentence":"until next time"})
# training_data.append({"class":"goodbye", "sentence":"it was nice seeing you"})
#
# training_data.append({"class":"goodbye", "sentence":"it's been real"})
# training_data.append({"class":"goodbye", "sentence":"im out of here"})
#
# training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
# training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
# training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
# training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})
#
# training_data.append({"class":"email", "sentence":"what's your email address?"})
# training_data.append({"class":"email", "sentence":"may I get your email?"})
# training_data.append({"class":"email", "sentence":"can I have your email?"})
# training_data.append({"class":"email", "sentence":"what's your email?"})
# training_data.append({"class":"email", "sentence":"let me get your email"})
# training_data.append({"class":"email", "sentence":"give me your email"})
# training_data.append({"class":"email", "sentence":"i'll take your email address"})
# training_data.append({"class":"email", "sentence":"can I have your business email?"})
# training_data.append({"class":"email", "sentence":"your email address?"})
# training_data.append({"class":"email", "sentence":"email please?"})
# training_data.append({"class":"email", "sentence":"may I have your email?"})
# training_data.append({"class":"email", "sentence":"can I get your email?"})


CLASS_WORDS = {}

CLASSES = list(set([a['class'] for a in training_data]))

CLASS_DICT = {}

LARGE_NEGATIVE_NUMBER = -10 # You cannot have log 0 in math or code

TRAINING_DATA_STATS = TrainingData("Training Data 1")

def initializeData():
    global CLASS_WORDS
    global CLASSES
    global CLASS_DICT
    global TRAINING_DATA_STATS
    global training_data
    global test_data

    ReadData.shuffleTrainingData()
    training_data = ReadData.TRAINING_DATA[:800]
    test_data = ReadData.TRAINING_DATA[800:]

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
                result = v * math.log10(numerator / denominator)
                doc.normalized_word_freq[k] = result

                # if k not in val.normalized_word_weight:
                #     val.normalized_word_weight[k] = doc.normalized_word_freq[k]
                # else:
                #     val.normalized_word_weight[k] += doc.normalized_word_freq[k]
                #
                # if k not in TRAINING_DATA_STATS.word_normalized_freq:
                #     TRAINING_DATA_STATS.word_normalized_freq[k] = doc.normalized_word_freq[k]
                # else:
                #     TRAINING_DATA_STATS.word_normalized_freq[k] += doc.normalized_word_freq[k]
                #
                # val.total_normalized_weight += result



def transformByLength():
    """
    Convert d_ij by dividing this value with the square root of
    The sum of the square of number of times a word exists in the document
    :return:
    """
    global CLASS_DICT
    global TRAINING_DATA_STATS

    for val in CLASS_DICT.values():
        for doc in val.documents:
            denominator = doc.getTotalNormalizedFreqByLength()
            for k, v in doc.normalized_word_freq.iteritems():
                result = v / denominator
                doc.normalized_word_freq[k] = result
                # print "Class: ", val.class_name, "Word: ", k, "Freq: ",doc.normalized_word_freq[k]

                if k not in val.normalized_word_weight:
                    val.normalized_word_weight[k] = doc.normalized_word_freq[k]
                else:
                    val.normalized_word_weight[k] += doc.normalized_word_freq[k]

                if k not in TRAINING_DATA_STATS.word_normalized_freq:
                    TRAINING_DATA_STATS.word_normalized_freq[k] = doc.normalized_word_freq[k]
                else:
                    TRAINING_DATA_STATS.word_normalized_freq[k] += doc.normalized_word_freq[k]

                val.total_normalized_weight += result


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

    for key, val in TRAINING_DATA_STATS.word_normalized_freq.iteritems():
        numerator = 0
        denominator = 0
        for c in CLASSES:
            # total sum - all instances of word found in class
            wordNormalizedFrequency = 0
            wordFrequency = 0
            if key in CLASS_DICT[c].normalized_word_weight:
                wordNormalizedFrequency = CLASS_DICT[c].normalized_word_weight[key]
                wordFrequency = CLASS_DICT[c].word_freq[key]

            # print key, " Total Freq ", val
            # print "Class Freq: ", wordFrequency

            numerator = ((val + TRAINING_DATA_STATS.word_freq[key] * alpha) - (wordNormalizedFrequency + wordFrequency * alpha))

            # print "Normalized Freq: ", TRAINING_DATA_STATS.getTotalNormalizedFreq()
            # print "Total Normalized Weight: ",CLASS_DICT[c].total_normalized_weight
            denominator = (TRAINING_DATA_STATS.getTotalNormalizedFreq() - CLASS_DICT[c].total_normalized_weight) * 1.0 + \
                          (TRAINING_DATA_STATS.word_freq[key] * alpha - wordFrequency * alpha)

            # print "Numerator: ", numerator
            # print "Denominator: ", denominator
            CLASS_DICT[c].normalized_word_weight[key] = numerator/denominator

def setWordWeights():
    """
    Pass all the weights after normalized and do a conversion.
    :return: Store new value into training data's word weight
    """
    global CLASS_DICT

    for c in CLASSES:
        for key, val in CLASS_DICT[c].normalized_word_weight.iteritems():
            if val != 0:
                CLASS_DICT[c].normalized_word_weight[key] = math.log10(val)
            else:
                CLASS_DICT[c].normalized_word_weight[key] = LARGE_NEGATIVE_NUMBER

def normalizeWordWeights():
    """
    Normalize the word weights stored from training data
    :return: The updated values of word weights
    """
    global CLASS_DICT

    for c in CLASSES:
        total = CLASS_DICT[c].getTotalWordWeight()
        for key, val in TRAINING_DATA_STATS.word_weight.iteritems():
            CLASS_DICT[c].normalized_word_weight[key] = val / total

    # for key, val in TRAINING_DATA_STATS.word_weight.iteritems():
    #     print key, ": ", val

def calculateClassScore(sentence, class_name, show_details=True):
    """
    Calculate the score of the class_name based on the sentence passed in.
    :param sentence: The sentence that will be scored
    :param class_name: The name of the class
    :param show_details: To show more information
    :return: The score of the class
    """
    "One thing to note. People suck at typing"
    global CLASS_DICT
    score = 0

    try:
        sentence = nltk.word_tokenize(sentence)
    except:
        pass

    # for word in nltk.word_tokenize(sentence):
    for word in sentence:
        try:
            word = stemmer.stem(word.lower())
        except:
            pass

        if word in CLASS_DICT[class_name].normalized_word_weight:
            # Treat each word with relative weight Times word frequency
            current_score = CLASS_DICT[class_name].normalized_word_weight[word]
            score += current_score

            if show_details:
                print (class_name,
                "   match: %s (%s)" % (word, current_score))
    return score

def classifyTWCNB(sentence):
    """
    Label a sentence using Transformed Weighted Compliment Naive Bayes
    :param sentence: The sentence that is being classfied
    :return: The best label and the lowest score
    """
    global CLASS_DICT

    best_class = None
    low_score = 100
    # loop through our classes
    for c in CLASS_DICT.keys():
        # calculate score of sentence for each class
        score = calculateClassScore(sentence, c, show_details=False)
        # keep track of highest score
        if score < low_score:
            best_class = c
            low_score = score

    return best_class, low_score

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
        score = calculate_class_score(sentence, c, show_details=False)
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score

    return high_class, high_score
#####################################################

######## ACCURACY TEST ################

def getAccuracy(test_data, show_details= True):
    """
    Find the accuracy of the classifier by feed it sentences from test_data
    :param test_data: List of sentences with classifications
    :return: The percent classified correctly
    """
    initializeData()
    total = len(test_data)
    correct = 0

    for data in test_data:
        high_class = classify(data['sentence'])[0]
        if high_class == data['class']:
            correct += 1
        if show_details == True:
            print "High Class: ", high_class, " ", type(high_class), " ", len(high_class)
            print "Correct Class: ", data['class'], " ", type(data['class']), " ", len(data['class'])

    return (correct * 1.0)/total

def getAverageAccuracy(repeats, test_data):
    """
    Run the accuracy after "repeats" amount of time and get the average
    :param repeats: The total number of times to run get Accuracy
    :param test_data: The data being fed into accuracy
    :return: The averaged percentage
    """
    totalAccuracy = 0
    count = 0

    while count < repeats:
        totalAccuracy += getAccuracy(test_data,show_details=False)
        count += 1
    return totalAccuracy/repeats


def getAccuracyTWCNB(test_data, show_details= True):
    """
    Find the accuracy of the classifier by feed it sentences from test_data
    :param test_data: List of sentences with classifications
    :return: The percent classified correctly
    """
    initializeData()
    transformTermFrequency()
    transformByDocFrequency()
    transformByLength()
    skewDataBiasHandler()
    setWordWeights()
    normalizeWordWeights()

    total = len(test_data)
    correct = 0

    for data in test_data:
        high_class = classifyTWCNB(data['sentence'])[0]
        if high_class == data['class']:
            correct += 1
        if show_details == True:
            print "High Class: ", high_class, " ", type(high_class), " ", len(high_class)
            print "Correct Class: ", data['class'], " ", type(data['class']), " ", len(data['class'])
    # print "REALLY THAT WAS IT??"
    return (correct * 1.0)/total

def getAverageAccuracyTWCNB(repeats, test_data):
    """
    Run the accuracy after "repeats" amount of time and get the average
    :param repeats: The total number of times to run get Accuracy
    :param test_data: The data being fed into accuracy
    :return: The averaged percentage
    """
    totalAccuracy = 0
    count = 0

    while count < repeats:
        accuracy = getAccuracyTWCNB(test_data,show_details=False)
        # print count, ": ", accuracy
        totalAccuracy += accuracy
        count += 1
    return totalAccuracy/repeats

#######################################

if __name__ == "__main__":
    # initializeData()
    # transformTermFrequency()
    # transformByDocFrequency()
    # transformByLength()
    # skewDataBiasHandler()
    # setWordWeights()
    # normalizeWordWeights()
    #
    # print classifyTWCNB("This was bad. How can something good be so bad?")
    #
    #
    # for values in CLASS_DICT.values():
    #     print values.class_name
    #     for k,v in values.normalized_word_weight.iteritems():
    #         print k, ": ", v
    #
    # print classifyTWCNB("It was a fun experience")
    # sentence = "Hello how are you doing?"
    # print classifyTWCNB(sentence)


    # Test using multinomial naive bayes
    # sentence = "Hello how are you doing?"
    #
    # print classify(sentence)


    ###### TEST MULTINOMIAL NAIVE BAYES
    # print getAverageAccuracy(10, test_data)

    ###### TEST TWC-Naive-Bayes

    print getAverageAccuracyTWCNB(10, test_data)


    ###### TEST REGULAR NAIVE BAYES
    # initializeData()
    # all_words = set(word.lower() for doc in training_data for word in nltk.word_tokenize(doc['sentence']))
    # featuresets = [({word: (word in nltk.word_tokenize(x['sentence'])) for word in all_words}, x['class']) for x in training_data]
    #
    # all_words = set(word.lower() for doc in test_data for word in nltk.word_tokenize(doc['sentence']))
    # testingsets = [({word: (word in nltk.word_tokenize(x['sentence'])) for word in all_words}, x['class']) for x in test_data]
    #
    # classifier = nltk.NaiveBayesClassifier.train(featuresets)
    # print nltk.classify.accuracy(classifier, testingsets)
