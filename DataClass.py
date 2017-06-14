"""

Class that keeps track of all word's frequency in the training_data, number of words altogether

"""

class TrainingData:
    def __init__(self, class_name):
        self.class_name = class_name
        self.word_freq = {}
        self.word_count = 0
        self.training_docs = []

    def addToTotalWordFreq(self, word):
        if word not in self.word_freq:
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def addWords(self, word):
        self.word_count += 1
