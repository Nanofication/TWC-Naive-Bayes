"""

The entire training set with the associated statistics for:
1. Total Word Frequency
2. Number of documents
3. Documents

"""

class TrainingSetStats:
    def __init__(self, class_name):
        self.class_name = class_name
        self.word_freq = {}
        self.word_normalized_freq = {}
        self.words = []
        self.word_count = 0
        self.training_docs = []
        self.word_weight = {}
        self.total_normalized_freq = 0

    def addToWordFreq(self, word):
        """
        Add the word to word_freq dictionary. If the word does not exist, add it to the dictionary with a value of 1
        :param word: The word added to the word frequency
        """
        if word not in self.word_freq:
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def addWords(self, word):
        """
        Add word to the word list and increment the total word count
        :param word: The word added to the list
        """
        self.words.extend(word)
        self.word_count += 1

    def getTotalWordWeight(self):
        total = 0
        for key, val in self.word_weight.iteritems():
            total += val

        self.totalWordWeight = total
        return total

    def getTotalFreq(self):
        total = 0
        for values in self.word_freq.values():
            total += values
        return total

    def getTotalNormalizedFreq(self):
        total = 0
        for values in self.word_normalized_freq.values():
            total += values
        return total

