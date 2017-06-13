"""

Class that stores each training data's class, sentence and times word occurs

"""


class Class:
    def __init__(self, class_name):
        self.class_name = class_name
        self.words = []
        self.word_freq = {}

    def addToWordFreq(self, word):
        if word not in self.word_freq:
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def addWords(self, word):
        self.words.extend(word)