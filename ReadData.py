import re
import random
import pandas as pd

TRAINING_DATA = []
DATA = {}

with open("amazon_cells_labelled.txt") as f:
    pattern = re.compile('[^A-Za-z0-9 \n]+')
    for line in f:
        line = line.rstrip().split("\t")

        TRAINING_DATA.append({'class':line[1].strip(), 'sentence':line[0].strip()})
    DATA['raw'] = TRAINING_DATA

def shuffleTrainingData():
    global TRAINING_DATA
    random.shuffle(TRAINING_DATA)


def readCSV():
    df = pd.read_csv('Sentiment_Analysis_Dataset.csv', nrows=20 , names = ['Sentiment','SentimentText'])
    df

# df = pd.read_csv('Sentiment_Analysis_Dataset.csv', nrows=5000)
#
# for index, row in df.iterrows():
#     TRAINING_DATA.append({'class':row['Sentiment'], 'sentence':row['SentimentText']})
