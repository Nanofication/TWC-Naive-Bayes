"""

The driver script that puts everything together

"""

import ReadData
import random

# SKLEARN Stuff
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories, shuffle=True, random_state=42)

twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories, shuffle=True, random_state=42)


from TWCNB_Class import TWC_Naive_Bayes as twcnb


"""
V2 Approach
- Build the TWCB layer for specific parsing for the class
- Layer 1, find the class (Maybe add a confidence layer)
- Layer 2, find the specific question or 3 questions related to the class.
    - We will load different data sets each with the class as the question that you want answered matched to
    the question closely associated to it.

Multi-layered Transformed Weight-Normalized Complimented Naive Bayes

"""

def runTWCNBLayer(sentence, csv_file_name, threshold, num_questions, force_one_output = False):
    """
    Set up TWCNB Layer and run it.
    1. Read the csv_file_name, shuffle and load the training data
    2. Create an instance of Transformed Weight-Normalized Complimented Naive Bayes and run it
    3. Return the 1 best classification or a set number of questions if the matched result crosses the threshold and
    one output is not forced.
    :param csv_file_name: The file name with the training data
    :param threshold: The percentage that is considered acceptable to return only 1 output
    :param num_questions: Number of questions to return
    :param force_one_output: If the layer will return only one output
    :return: Return either 1 class_name or a list of potential class_names
    """
    ReadData.clearTrainingData()
    ReadData.readCSV(csv_file_name)

    ReadData.shuffleTrainingData()
    training_data = ReadData.TRAINING_DATA

    twcb = twcnb(training_data, test_training_set=False)
    best_classification = twcb.classifyTWCNB(sentence, classifying_sentence=True, show_details=False)

    if force_one_output == True:
        return best_classification[0]

    return twcb.calculateMatchedSentencePercent(sentence, threshold, num_questions, debug=False)


if __name__ == "__main__":
    # Weather.csv has 5 classes and 10 questions
    # Traffic.csv has 2 classes and 10 questions

    # We can do a match calculation
    sentence = "How's the weather today man?"
    # sentence = "Can I get a traffic report" # <-- Got a little lucky on this one in my opinion. Not enough data sets.

    best_class = runTWCNBLayer(sentence, 'AI_Questions_Sheet1', 50, 3, force_one_output=True) # Layer 1

    output = runTWCNBLayer(sentence, best_class, 40, 3) # Layer 2

    print output

    ###### FREQUENCY #######
    #
    # best_class = getHighestClass(classifications)
    # print best_class
    #
    # docs = twcb.bestFitQuestions(best_class, sentence, 3)
    #
    # for x in docs:
    #     print x["sentence"]

    # best_classification = twcb.classifyTWCNB(sentence) # Classification is a little choppy
    #
    # print twcb.bestFitQuestions(best_classification[0], "How many deals did we close?", 3)
    #
    ###### ACCURACY TEST ####### Remember to use openFile() function for Amazon dummy data
    #
    # ReadData.readCSV("AI_Questions_Dataset")
    # # ReadData.readCSV('AI_Questions_Dataset')
    # ReadData.shuffleTrainingData()
    # TRAINING_DATA = ReadData.TRAINING_DATA
    #
    # twcb = twcnb(TRAINING_DATA, True)
    # print twcb.getAverageAccuracyTWCNB(100, False, False)




    ###### SKLearn Dataset ######
    # Current TWCNB is too slow to calculate this.
    # for i in range(len(twenty_train.data)):
    #     ReadData.addToTrainingData(twenty_train.target[i], twenty_train.data[i])
    #
    # for val in ReadData.TRAINING_DATA:
    #     print val
    #
    # print len(ReadData.TRAINING_DATA)

    # ReadData.shuffleTrainingData()
    # TRAINING_DATA = ReadData.TRAINING_DATA
    #
    # twcb = twcnb(TRAINING_DATA[:1000], True)
    # print twcb.getAverageAccuracyTWCNB(10, False, False)

    # ReadData.addToTrainingData()

    # target_names
    # data
    # target