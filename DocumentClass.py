"""

Keeps track of individual statistics of a document fed into the algorithm

"""

class Document:
    def __init__(self, doc_num, data):
        self.doc_num = doc_num + 1
        self.data = data
        self.word_freq = {}

    def parseDataAddWordFreq(self):
        for word in self.data:
            self.addToDocsWordFreq(word)

    def addToDocsWordFreq(self, word):
        if word not in self.word_freq:
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1
