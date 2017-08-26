import re
import random
import pandas as pd

TRAINING_DATA = []
DATA = {}

#with open("amazon_cells_labelled.txt") as f:

# with open("amazon_cells_labelled.txt") as f:
#     pattern = re.compile('[^A-Za-z0-9 \n]+')
#     for line in f:
#         line = line.rstrip().split("\t")
#         TRAINING_DATA.append({'class': line[1].strip(), 'sentence': line[0].strip()})
#     DATA['raw'] = TRAINING_DATA


def openFile():
    with open("amazon_cells_labelled.txt") as f:
        pattern = re.compile('[^A-Za-z0-9 \n]+')
        for line in f:
            line = line.rstrip().split("\t")
            TRAINING_DATA.append({'class':line[1].strip(), 'sentence':line[0].strip()})
        DATA['raw'] = TRAINING_DATA

def shuffleTrainingData():
    global TRAINING_DATA
    random.shuffle(TRAINING_DATA)


def readCSV(csv_file_name):
    #AI_Questions_Dataset
    df = pd.read_csv(csv_file_name + '.csv', names=['Class', 'Sentence'], error_bad_lines=False)

    for index, row in df.iterrows():
        TRAINING_DATA.append({'class': row['Class'], 'sentence': row['Sentence']})

def addToTrainingData(class_name, sentence):
    global TRAINING_DATA
    TRAINING_DATA.append({'class': class_name, 'sentence': sentence})

def clearTrainingData():
    global TRAINING_DATA
    TRAINING_DATA = []