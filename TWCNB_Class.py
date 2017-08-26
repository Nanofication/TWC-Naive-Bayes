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

import re
import math
import nltk
from nltk.stem.lancaster import LancasterStemmer

from Classification import Classification
from TrainingSetStats import TrainingSetStats
from Document import Document

# Word stemmer. Reduce words to the root forms for better classification
stemmer = LancasterStemmer()

LARGE_NEGATIVE_NUMBER = -99999 # You cannot have log 0 in math or code. It approaches negative infinity.

class TWC_Naive_Bayes:
    def __init__(self, data_set, test_training_set= True):
        self.data_set = data_set
        self.classifications = list(set([a['class'] for a in data_set]))
        self.test_training_data = test_training_set


    def initializeData(self):
        self.class_dict = {}
        self.training_data_stats = TrainingSetStats("Training Set")
        self.training_data = []
        self.test_data = []
        self.class_matched = {}

        random.shuffle(self.data_set)
        training_data_amount = len(self.data_set)

        if self.test_training_data == True:
            # We follow train with 80%, test with 20%
            training_data_amount = training_data_amount * 4/5
        self.training_data = self.data_set[:training_data_amount]
        self.test_data = self.data_set[training_data_amount:]

        for i in range(len(self.classifications)):
            self.class_dict[self.classifications[i]] = Classification(self.classifications[i])

        for data in self.training_data:
            raw_data = data
            # Tokenize each sentence into words
            sentence = self.removeSpecialCharacters(data['sentence'])
            sentence = nltk.word_tokenize(sentence)
            sentence = self.lemmatizeSentence(sentence)
            doc_num = self.class_dict[data['class']].getTotalDocuments()

            document = Document(doc_num, data['class'], raw_data, sentence)
            self.class_dict[data['class']].addDocument(document)
            self.training_data_stats.training_docs.append(sentence)

            for word in sentence:
                # Have we not seen this word already?
                self.class_dict[data['class']].addToWordFreq(word)
                # Add the word to our words in class list
                self.class_dict[data['class']].documents[doc_num].addToWordFreq(word)

                # Add words to each class's list and increment the word count
                self.class_dict[data['class']].addWords([word])

                self.training_data_stats.addToWordFreq(word)
                self.training_data_stats.addWords([word])

    ####### HELPER FUNCTIONS ############
    def lemmatizeSentence(self, sentence):
        """
        Reduce each word in the sentence to its base case
        :param sentence: The entire sentence that will be reduced to its base case
        :return: The sentence with words in its base case
        """
        for i in range(len(sentence)):
            sentence[i] = stemmer.stem(sentence[i])
        return sentence

    def removeSpecialCharacters(self, sentence):
        pattern = re.compile('[^A-Za-z0-9 ]+')
        sentence = re.sub(pattern, '', sentence)
        return sentence

    def wordInDocument(self, word, sentence):
        """
        Check if the word passed in is in the document.
        :param word: The word being checked if the document contains it
        :return: If word exists in document return 1 else 0
        """
        if word in sentence:
            return 1
        return 0

    def checkAllInstancesOfWord(self, word):
        """
        Check the number of times the word occurs in the list of training documents
        :param word: Word to be checked in each document
        :return: Number of docs with one or more instance of word
        """
        total = 0
        for doc in self.training_data_stats.training_docs:
            total += self.wordInDocument(word, doc)
        return total

    ###### Transformed Weighted Complement Naive Bayes #####

    def transformTermFrequency(self):
        """
        Transform all word weights from each class's individual documents
        using the equation d_ij = log(d_ij + 1)

        Return class with updated weights
        """
        for classification in self.class_dict.values():
            for doc in classification.documents:
                doc.normalizeWordFreq()

    def inverseDocFrequencyTransform(self):
        """
        Transform the word weight based on frequency of the word occurring
        in all documents.

        Caution: Runtime O(n^4) Though not that many keys to cause a problem

        Follow this equation d_ij = d_ij * log((Sum of all docs)/ Sum of if word occurs in doc)
        :return: The class with updated weights.
        """
        numerator = len(self.training_data_stats.training_docs)
        for classification in self.class_dict.values():
            for doc in classification.documents:
                for word, freq in doc.normalized_word_freq.iteritems():
                    denominator = self.checkAllInstancesOfWord(word)
                    base = math.log10(numerator / denominator)
                    result = freq * base
                    doc.normalized_word_freq[word] = result

    def transformByLength(self):
        """
        Convert d_ij by dividing this value with the square root of
        The sum of the square of number of times a word exists in the document
        """
        for classification in self.class_dict.values():
            for doc in classification.documents:
                denominator = doc.getTotalNormalizedFreqByLength()

                for word, freq in doc.normalized_word_freq.iteritems():
                    result = freq / denominator
                    doc.normalized_word_freq[word] = result

                    if word not in classification.normalized_word_weight:
                        classification.normalized_word_weight[word] = doc.normalized_word_freq[word]
                    else:
                        classification.normalized_word_weight[word] += doc.normalized_word_freq[word]

                    if word not in self.training_data_stats.word_normalized_freq:
                        self.training_data_stats.word_normalized_freq[word] = doc.normalized_word_freq[word]
                    else:
                        self.training_data_stats.word_normalized_freq[word] += doc.normalized_word_freq[word]

                    classification.total_normalized_weight += result

        self.training_data_stats.total_normalized_freq = self.training_data_stats.getTotalNormalizedFreq()

    def skewDataBiasHandler(self):
        """
        Set the parameter weights of all words into a less biased value

        Use the equation:
        Normalized frequency = (SUM of specific word's frequency in all training sets - word frequency in current class)
        Divided by (SUM of all word's frequency - words frequency in the given class)
        """
        alpha = 1

        for word, freq in self.training_data_stats.word_normalized_freq.iteritems():
            numerator = 0
            denominator = 0
            for classification in self.classifications:
                # total sum - all instances of word found in class
                wordNormalizedFrequency = 0.0
                wordFrequency = 0
                if word in self.class_dict[classification].normalized_word_weight:
                    wordNormalizedFrequency = self.class_dict[classification].normalized_word_weight[word]
                    wordFrequency = self.class_dict[classification].word_freq[word]

                numerator = ((freq + self.training_data_stats.word_freq[word] * alpha) -
                             (wordNormalizedFrequency + wordFrequency * alpha))


                denominator = ((self.training_data_stats.total_normalized_freq - self.class_dict[classification].total_normalized_weight) * 1.0 +
                               (self.training_data_stats.word_freq[word] * alpha - wordFrequency * alpha))

                self.class_dict[classification].normalized_word_weight[word] = numerator / denominator

        self.training_data_stats.word_normalized_freq = {}

    def setWordWeights(self):
        """
        Set all the weights after normalized and do a conversion with log.
        If the normalized weight is 0, set it to a large negative number.
        :return: Re-Store new value into training data's normalized word weight
        """
        for c in self.classifications:
            for word, freq in self.class_dict[c].normalized_word_weight.iteritems():
                if freq != 0:
                    self.class_dict[c].normalized_word_weight[word] = math.log10(freq)
                else:
                    self.class_dict[c].normalized_word_weight[word] = LARGE_NEGATIVE_NUMBER

                if word not in self.training_data_stats.word_normalized_freq:
                    self.training_data_stats.word_normalized_freq[word] = self.class_dict[c].normalized_word_weight[word]
                else:
                    self.training_data_stats.word_normalized_freq[word] += self.class_dict[c].normalized_word_weight[word]

        self.training_data_stats.total_normalized_freq = self.training_data_stats.getTotalNormalizedFreq()

    def normalizeWordWeights(self):
        """
        Normalize the word weights stored from training data. Denominator are the weights of that word in all documents
        :return: The updated values of word weights
        """
        for c in self.classifications:
            for word, freq in self.training_data_stats.word_normalized_freq.iteritems():
                numerator = self.class_dict[c].normalized_word_weight[word]
                denominator = math.fabs(freq)

                self.class_dict[c].normalized_word_weight[word] = numerator / denominator

    def calculateClassScore(self, sentence, class_name, show_details=True):
        """
        Calculate the score of the class_name based on the sentence passed in.
        :param sentence: The sentence that will be scored
        :param class_name: The name of the class
        :param show_details: To show more information
        :return: The score of the class
        """
        "One thing to note. People suck at typing"
        score = 0

        try:
            sentence = nltk.word_tokenize(sentence)
        except:
            pass
        for word in sentence:
            try:
                word = stemmer.stem(word.lower())
            except:
                pass
            if word in self.class_dict[class_name].normalized_word_weight:
                # Treat each word with relative weight Times word frequency
                current_score = self.class_dict[class_name].normalized_word_weight[word]
                score += current_score

                if show_details == True:
                    print (class_name,
                    "   match: %s (%s)" % (word, current_score))
        if show_details == True:
            print class_name, " score: ", score

        self.class_matched[class_name] = score

        return score

    def classifyTWCNB(self, sentence, classifying_sentence = False, show_details=False):
        """
        Label a sentence using Transformed Weighted Compliment Naive Bayes
        :param sentence: The sentence that is being classfied
        :return: The best label and the lowest score
        """
        if classifying_sentence == True:
            self.initializeData()
            self.transformTermFrequency()
            self.inverseDocFrequencyTransform()
            self.transformByLength()
            self.skewDataBiasHandler()
            self.setWordWeights()
            self.normalizeWordWeights()

        best_class = None
        low_score = 100
        # loop through our classes
        for c in self.class_dict.keys():
            # calculate score of sentence for each class
            score = self.calculateClassScore(sentence, c, show_details)
            # keep track of highest score
            if score < low_score:
                best_class = c
                low_score = score

        return best_class, low_score

    ####### MULTINOMIAL NAIVE BAYES TEST CODE. We use this as a benchmark #######

    def calculate_class_score(self, sentence, class_name, show_details=True):
        score = 0

        for word in nltk.word_tokenize(sentence):
            if stemmer.stem(word.lower()) in self.class_dict[class_name].word_freq:
                # Treat each word with relative weight
                current_score = 1.0 / self.training_data_stats.word_freq[stemmer.stem(word.lower())]
                score += current_score

                if show_details:
                    print (
                        "   match: %s (%s)" % (
                        stemmer.stem(word.lower()), 1.0 / self.training_data_stats.word_freq.word_freq[stemmer.stem(word.lower())]))
        return score

    def classify(self, sentence):
        high_class = None
        high_score = 0
        # loop through our classes
        for c in self.class_dict.keys():
            # calculate score of sentence for each class
            score = self.calculate_class_score(sentence, c, show_details=False)
            # keep track of highest score
            if score > high_score:
                high_class = c
                high_score = score

        return high_class, high_score

    ######## ACCURACY TEST ################

    def getAccuracyMultinomial(self, show_details= True):
        """
        Find the accuracy of the classifier by feed it sentences from test_data
        :param test_data: List of sentences with classifications
        :return: The percent classified correctly
        """
        self.initializeData()
        total = len(self.test_data)
        correct = 0

        for data in self.test_data:
            high_class = self.classify(data['sentence'])[0]
            if high_class == data['class']:
                correct += 1
            if show_details == True:
                print "High Class: ", high_class, " ", type(high_class), " ", len(high_class)
                print "Correct Class: ", data['class'], " ", type(data['class']), " ", len(data['class'])
        return (correct * 1.0)/total

    def getAverageAccuracyMultinomial(self, repeats, show_details=False):
        """
        Run the accuracy after "repeats" amount of time and get the average
        :param repeats: The total number of times to run get Accuracy
        :return: The averaged percentage
        """
        totalAccuracy = 0
        count = 0

        while count < repeats:
            accuracy = self.getAccuracyMultinomial(show_details)
            # print count, ": ", accuracy
            totalAccuracy += accuracy
            count += 1
        return totalAccuracy/repeats


    def getAccuracyTWCNB(self, show_details= True, show_breakdown= False):
        """
        Find the accuracy of the classifier by feed it sentences from test_data
        :param test_data: List of sentences with classifications
        :return: The percent classified correctly
        """
        self.initializeData()
        self.transformTermFrequency()
        self.inverseDocFrequencyTransform()
        self.transformByLength()
        self.skewDataBiasHandler()
        self.setWordWeights()
        self.normalizeWordWeights()

        total = len(self.test_data)
        correct = 0

        for data in self.test_data:
            high_class = self.classifyTWCNB(data['sentence'], show_details=show_breakdown)[0]
            if high_class == data['class']:
                correct += 1
            if show_details == True:
                print data['sentence']
                print "High Class: ", high_class, " ", type(high_class), " ", len(high_class)
                print "Correct Class: ", data['class'], " ", type(data['class']), " ", len(data['class'])
                print "_______________________________________"
        return (correct * 1.0)/total

    def getAverageAccuracyTWCNB(self, repeats, show_details=False, show_breakdown= False):
        """
        Run the accuracy after "repeats" amount of time and get the average
        :param repeats: The total number of times to run get Accuracy
        :return: The averaged percentage
        """
        totalAccuracy = 0
        count = 0

        while count < repeats:
            accuracy = self.getAccuracyTWCNB(show_details, show_breakdown)
            print count, ": ", accuracy
            totalAccuracy += accuracy
            count += 1
        return totalAccuracy/repeats

           # Testing purposes #
    ###### MATCHING PERCENTAGE ######

    def calculateMatchedSentencePercent(self, sentence, threshold, number_of_questions, debug=False):
        total_score = 0
        highest_percentage = 0.0
        best_class = None

        for score in self.class_matched.values():
            total_score += score

        if total_score == 0:
            return "Please ask a question related to sales and marketing."

        for class_name, score in self.class_matched.iteritems():
            current_percentage = score/total_score

            if highest_percentage < current_percentage:
                highest_percentage = current_percentage
                best_class = class_name
            if debug == True:
                print class_name,": ", current_percentage * 100

        highest_percentage = highest_percentage * 100

        return self.returnBestQuestion(best_class, sentence, highest_percentage, threshold, number_of_questions)

    def returnBestQuestion(self, class_name, sentence, percentage, threshold, number_of_questions):
        """
        Send back 1 question if the highest percentage of the given class_name and sentence goes over
        a certain threshold. Otherwise, return the top number of questions based on frequency of matched words.
        :param class_name: Best class
        :param sentence: The sentence being parsed
        :param percentage: Highest matched percentage of all words
        :param threshold: The accepted percentage
        :param number_of_questions: The number of questions to be returned if the percentage is less than the threshold
        """
        if percentage >= threshold:
            return class_name
        return self.bestFitQuestions(class_name, sentence, number_of_questions)


    ###### Get Top Number of Questions by Word Frequency #######

    def bestFitQuestions(self, classification, sentence, number):
        """
        Within the classification Calculate each question's score within the document based on the frequency of each
        word within the given sentence. And send back the top ranked "number" of questions.
        :return: The top set of questions
        """
        scoredDocs = []
        topDocs = []

        sentence = self.removeSpecialCharacters(sentence)
        sentence = nltk.word_tokenize(sentence)
        sentence = self.lemmatizeSentence(sentence)

        for doc in self.class_dict[classification].documents:
            score = 0
            for word in sentence:
                if word in doc.word_freq:
                    score += doc.word_freq[word]
            scoredDocs.append({"document":doc, "score": score})

        scoredDocs = self.sort(scoredDocs)

        for i in range(len(scoredDocs), len(scoredDocs) - number, -1):
            topDocs.append(scoredDocs[i-1]["document"].raw_data["sentence"])

        return topDocs

    def partition(self, lst, start, end, pivot):
        """
        :param lst: The list being partitioned. The list frequency is the second index per item
        :param start: Start of the list
        :param end: End of the list
        :param pivot: An arbitrary point between the start and end
        :return:
        """
        lst[pivot], lst[end] = lst[end], lst[pivot]
        store_index = start
        for i in xrange(start,end):
            if lst[i]["score"] < lst[end]["score"]:
                lst[i], lst[store_index] = lst[store_index], lst[i]
                store_index += 1
        lst[store_index], lst[end] = lst[end], lst[store_index]
        return store_index



    def quickSort(self, unsortedList, start, end):
        """
        Quick sort the list of best fit questions
        Runtime should be O(nlogn)
        :param start: The beginning of the list
        :param end: End of the list
        :return: The sorted list
        """
        if start >= end:
            return unsortedList
        pivot = random.randrange(start, end + 1)
        new_pivot = self.partition(unsortedList, start, end, pivot)

        self.quickSort(unsortedList, start, new_pivot - 1)
        self.quickSort(unsortedList, new_pivot + 1, end)

    def sort(self, lst):
        self.quickSort(lst, 0, len(lst) - 1)
        return lst
