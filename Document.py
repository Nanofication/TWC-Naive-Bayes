"""

The phrase sentence or paragraph with the associated classification.


In the case of text, remember to pass in sentences with lemmatized words
"""

import math

class Document:
    def __init__(self, doc_num, classification, raw_data, data):
        self.doc_num = doc_num + 1
        self.classification = classification
        self.raw_data = raw_data
        self.data = data
        self.word_freq = {}
        self.normalized_word_freq = {}

    def parseDataAddWordFreq(self):
        for word in self.data:
            self.addToWordFreq(word)

    def addToWordFreq(self, word):
        if word not in self.word_freq:
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def normalizeWordFreq(self):
        """
        Transform the word weights in word_freq. d_ij is the frequency of the word
        using the equation d_ij = log(d_ij + 1)

        Return class with updated weights
        """
        for word, freq in self.word_freq.iteritems():
            self.normalized_word_freq[word] = math.log10(freq + 1)
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
