"""

Keeps track of individual statistics of a document fed into the algorithm

"""
import math

class Document:
    def __init__(self, doc_num, classification, data):
        self.doc_num = doc_num + 1
        self.classification = classification
        self.data = data
        self.word_freq = {}
        self.normalized_word_freq = {}

    def parseDataAddWordFreq(self):
        for word in self.data:
            self.addToDocsWordFreq(word)

    def addToDocsWordFreq(self, word):
        if word not in self.word_freq:
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def normalizeWordFreq(self):
        for key, val in self.word_freq.iteritems():
            self.normalized_word_freq[key] = math.log10(val + 1)

    def sumNormalizedFreq(self):
        sumFreq = 0

        for key, val in self.normalized_word_freq.iteritems():
            sumFreq += val
        self.sum_normalized_freq = sumFreq

        return sumFreq

    def getTotalNormalizedFreqByLength(self):
        """
        Calculate the total frequencies in the document using the equation
        by squaring each individual normalized frequency, summing them and square rooting the entire equation
        :return: The solution after passing in the equation.
        """
        solution = 0.0
        for frequency in self.normalized_word_freq.values():
            solution += frequency**2

        return math.sqrt(solution)

