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
import nltk
from nltk.stem.lancaster import LancasterStemmer

from Classification import Classification
from TrainingSetStats import TrainingSetStats
from Document import Document

# Word stemmer. Reduce words to the root forms for better classification
stemmer = LancasterStemmer()

# TESTING PURPOSES
ReadData.shuffleTrainingData()
TRAINING_DATA = ReadData.TRAINING_DATA[:80]
TEST_DATA = ReadData.TRAINING_DATA[80:]

CLASSIFICATIONS = list(set([a['class'] for a in TRAINING_DATA]))

CLASS_DICT = {}

LARGE_NEGATIVE_NUMBER = -99999 # You cannot have log 0 in math or code. It approaches negative infinity.

TRAINING_DATA_STATS = TrainingSetStats("Training Set 1")

def initializeData():
    global CLASSIFICATIONS
    global CLASS_DICT
    global TRAINING_DATA_STATS
    global TRAINING_DATA
    global TEST_DATA

    ReadData.shuffleTrainingData()
    TRAINING_DATA = ReadData.TRAINING_DATA[:800]
    TEST_DATA = ReadData.TRAINING_DATA[80:]

    for i in range(len(CLASSIFICATIONS)):
        CLASS_DICT[CLASSIFICATIONS[i]] = Classification(CLASSIFICATIONS[i])

    for data in TRAINING_DATA:
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
            CLASS_DICT[data['class']].addToWordFreq(word)
            # Add the word to our words in class list
            CLASS_DICT[data['class']].documents[doc_num].addToWordFreq(word)

            # Add words to each class's list and increment the word count
            CLASS_DICT[data['class']].addWords([word])

            TRAINING_DATA_STATS.addToWordFreq(word)
            TRAINING_DATA_STATS.addWords([word])

####### HELPER FUNCTIONS ############
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
    for classification in CLASS_DICT.values():
        for doc in classification.documents:
            doc.normalizeWordFreq()

def inverseDocFrequencyTransform():
    """
    Transform the word weight based on frequency of the word occurring
    in all documents.

    Caution: Runtime O(n^4) Though not that many keys to cause a problem

    Follow this equation d_ij = d_ij * log((Sum of all docs)/ Sum of if word occurs in doc)
    :return: The class with updated weights.
    """
    global CLASS_DICT
    global TRAINING_DATA_STATS

    numerator = len(TRAINING_DATA_STATS.training_docs)

    for classification in CLASS_DICT.values():
        for doc in classification.documents:
            for word, freq in doc.normalized_word_freq.iteritems():
                denominator = checkAllInstancesOfWord(word)
                result = freq * math.log10(numerator / denominator)
                doc.normalized_word_freq[word] = result


def transformByLength():
    """
    Convert d_ij by dividing this value with the square root of
    The sum of the square of number of times a word exists in the document
    """
    global CLASS_DICT
    global TRAINING_DATA_STATS

    for classification in CLASS_DICT.values():
        for doc in classification.documents:
            denominator = doc.getTotalNormalizedFreqByLength()

            for word, freq in doc.normalized_word_freq.iteritems():
                result = freq / denominator
                doc.normalized_word_freq[word] = result
                # print "Class: ", val.class_name, "Word: ", k, "Freq: ",doc.normalized_word_freq[k]

                if word not in classification.normalized_word_weight:
                    classification.normalized_word_weight[word] = doc.normalized_word_freq[word]
                else:
                    classification.normalized_word_weight[word] += doc.normalized_word_freq[word]

                if word not in TRAINING_DATA_STATS.word_normalized_freq:
                    TRAINING_DATA_STATS.word_normalized_freq[word] = doc.normalized_word_freq[word]
                else:
                    TRAINING_DATA_STATS.word_normalized_freq[word] += doc.normalized_word_freq[word]

                classification.total_normalized_weight += result

    TRAINING_DATA_STATS.total_normalized_freq = TRAINING_DATA_STATS.getTotalNormalizedFreq()


def skewDataBiasHandler():
    """
    Set the parameter weights of all words into a less biased value

    Use the equation:
    Normalized frequency = (SUM of specific word's frequency in all training sets - word frequency in current class)
    Divided by (SUM of all word's frequency - words frequency in the given class)
    """

    global TRAINING_DATA_STATS
    global CLASS_DICT
    global CLASSIFICATIONS

    alpha = 1

    for word, freq in TRAINING_DATA_STATS.word_normalized_freq.iteritems():
        numerator = 0
        denominator = 0
        for classification in CLASSIFICATIONS:
            # total sum - all instances of word found in class
            wordNormalizedFrequency = 0
            wordFrequency = 0
            if word in CLASS_DICT[classification].normalized_word_weight:
                wordNormalizedFrequency = CLASS_DICT[classification].normalized_word_weight[word]
                wordFrequency = CLASS_DICT[classification].word_freq[word]

            # print key, " Total Freq ", val
            # print "Class Freq: ", wordFrequency

            numerator = ((freq + TRAINING_DATA_STATS.word_freq[word] * alpha) -
                         (wordNormalizedFrequency + wordFrequency * alpha))

            # print "Normalized Freq: ", TRAINING_DATA_STATS.getTotalNormalizedFreq()
            # print "Total Normalized Weight: ",CLASS_DICT[c].total_normalized_weight
            denominator = ((TRAINING_DATA_STATS.total_normalized_freq - CLASS_DICT[classification].total_normalized_weight) * 1.0 +
                          (TRAINING_DATA_STATS.word_freq[word] * alpha - wordFrequency * alpha))

            # print "Numerator: ", numerator
            # print "Denominator: ", denominator
            CLASS_DICT[classification].normalized_word_weight[word] = numerator/denominator

def setWordWeights():
    """
    Pass all the weights after normalized and do a conversion.
    :return: Store new value into training data's word weight
    """
    global CLASS_DICT

    for c in CLASSIFICATIONS:
        for word, freq in CLASS_DICT[c].normalized_word_weight.iteritems():
            if freq != 0:
                CLASS_DICT[c].normalized_word_weight[word] = math.log10(freq)
            else:
                CLASS_DICT[c].normalized_word_weight[word] = LARGE_NEGATIVE_NUMBER

def normalizeWordWeights():
    """
    Normalize the word weights stored from training data
    :return: The updated values of word weights
    """
    global CLASS_DICT

    for c in CLASSIFICATIONS:
        total = CLASS_DICT[c].getTotalWordWeight()
        for word, freq in TRAINING_DATA_STATS.word_weight.iteritems():
            CLASS_DICT[c].normalized_word_weight[word] = freq / total
            print word
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

######## ACCURACY TEST ################

def getAccuracyTWCNB(test_data, show_details= True):
    """
    Find the accuracy of the classifier by feed it sentences from test_data
    :param test_data: List of sentences with classifications
    :return: The percent classified correctly
    """
    initializeData()
    transformTermFrequency()
    inverseDocFrequencyTransform()
    transformByLength()
    skewDataBiasHandler()
    setWordWeights()
    # normalizeWordWeights()

    total = len(test_data)
    correct = 0

    for data in test_data:
        high_class = classifyTWCNB(data['sentence'])[0]
        if high_class == data['class']:
            correct += 1
        if show_details == True:
            print "High Class: ", high_class, " ", type(high_class), " ", len(high_class)
            print "Correct Class: ", data['class'], " ", type(data['class']), " ", len(data['class'])
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
        print count, ": ", accuracy
        totalAccuracy += accuracy
        count += 1
    return totalAccuracy/repeats

#######################################

if __name__ == "__main__":
    initializeData()
    transformTermFrequency()
    inverseDocFrequencyTransform()
    transformByLength()
    skewDataBiasHandler()
    setWordWeights()
    # normalizeWordWeights()
    # # If you are getting a really negative number, you just need to balance it a little with counter training data.
    # # Sophisticated requests may confuse the classifier
    print classifyTWCNB("What was last year's sales?") # Report

    ###### TEST TWC-Naive-Bayes Accuracy
    # initializeData()
    # print getAverageAccuracyTWCNB(10, test_data)


