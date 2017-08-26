"""

The classification that documents are labelled under with statistics on:

1. Each unique word's frequency in all the documents
2. All documents associated with this classification
3. The normalized frequency of all words
4. Total word count

"""

class Classification:
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

    def addDocument(self, document):
        """
        Add document to list of documents part of the classification
        :param document: Document added to the list
        """
        self.documents.append(document)

    def getTotalDocuments(self):
        return len(self.documents)

    def getTotalWordWeight(self):
        total = 0
        for val in self.normalized_word_weight.values():
            total += val
        return total