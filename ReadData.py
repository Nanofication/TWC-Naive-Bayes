import re
import pandas as pd

TRAINING_DATA = []
DATA = {}

with open("amazon_cells_labelled.txt") as f:
    pattern = re.compile('[^A-Za-z0-9 \n]+')
    for line in f:
        line = line.rstrip().split("\t")

        TRAINING_DATA.append({'class':line[1].strip(), 'sentence':line[0].strip()})
    DATA['raw'] = TRAINING_DATA

# for data in TRAINING_DATA:
#     print data
#
# print len(TRAINING_DATA)

# df = pd.DataFrame(DATA, columns = ['raw'])
#
# df['class'] = df['raw'].str.extract('([A-Z]\w{0,})', expand = True)
# print df['class']