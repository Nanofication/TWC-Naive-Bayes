"""

Class that stores each training data's class, sentence and times word occurs

"""

from DocumentClass import Document

class Class:
    def __init__(self, class_name):
        self.class_name = class_name
        self.words = []
        self.word_freq = {}
        self.word_count = 0
        self.total_training_docs = 0
        self.documents = []
        self.word_weight = {}
        self.normalized_word_weight = {}
        self.total_normalized_weight = 0

    def addToTotalClassWordFreq(self, word):
        if word not in self.word_freq:
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def addWords(self, word):
        self.words.extend(word)
        self.word_count += 1

    def addDocument(self, document):
        self.documents.append(document)

    def getTotalDocuments(self):
        return len(self.documents)

    def getTotalWordWeight(self):
        total = 0
        for val in self.normalized_word_weight.values():
            total += val
        return total



        self.totalWordWeight = total
        return total

