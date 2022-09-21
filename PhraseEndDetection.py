import toolbox
import csv
from toolbox import END_sequence_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
import pickle

#data_augmented = False
TOKENIZED_PIECES = pickle.load(open("TavernCorpus_TOKENIZED_PIECES.p", "rb" ))
authors = ["Mozart", "Beethoven"]
tokenizations = ['REMI', 'CPWord', 'PitchShift', 'PitchMute', 'SpatialPS']

train_size = 0.6
test_size = 1 - train_size

phrase_length = 2 # 2 mesures
nb_phrases = 100

nb_loop = 10
ngram_range = (1,1)

# Accuracy and Vocabulary dictionnaries, relative to each tokenization
ACCURACY_DIC = {}
VOCAB_DIC = {}
for tokenization in tokenizations:
    ACCURACY_DIC[tokenization] = [tokenization]
    VOCAB_DIC[tokenization] = [tokenization]

# Loop
for i in range(200, 200 + nb_loop):
    random_state = i

    CORPUS_DIC = {
        'REMI': [[], [], [], []], # X_train, X_test, y_train, y_test
        'CPWord': [[], [], [], []],
        'PitchShift': [[], [], [], []],
        'PitchMute': [[], [], [], []],
        'SpatialPS': [[], [], [], []]
    }

    rs = ShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=random_state)

    for author in authors:
        for tokenization in tokenizations:
            for train_index, test_index in rs.split(TOKENIZED_PIECES[author][tokenization]): 
                for indice in train_index:
                    phrases, END_presence_list = END_sequence_split(TOKENIZED_PIECES[author][tokenization][indice], phrase_length=phrase_length, nb_phrases=nb_phrases)
                    for phrase in phrases:
                        X_train_sequence = ''.join(str(token) + ' ' for token in phrase)
                        CORPUS_DIC[tokenization][0].append(X_train_sequence)

                    CORPUS_DIC[tokenization][2].extend([str(int(END_present)) for END_present in END_presence_list])

                for indice in test_index:
                    phrases, END_presence_list = END_sequence_split(TOKENIZED_PIECES[author][tokenization][indice], phrase_length=phrase_length, nb_phrases=nb_phrases)
                    for phrase in phrases:
                        X_test_sequence = ''.join(str(token) + ' ' for token in phrase)
                        CORPUS_DIC[tokenization][1].append(X_test_sequence)

                    CORPUS_DIC[tokenization][3].extend([str(int(END_present)) for END_present in END_presence_list])



    for tokenization in tokenizations:
        CORPUS = CORPUS_DIC[tokenization]
        X_train, X_test, y_train, y_test = CORPUS[0], CORPUS[1], CORPUS[2], CORPUS[3]

        # Initiate regressor and vectorizer
        regressor = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter=10000) # n_grams
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)

        # Process data
        X_train_processed = vectorizer.fit_transform(X_train)
        X_test_processed = vectorizer.transform(X_test)
        regressor = regressor.fit(X_train_processed, y_train)
        y_pred = regressor.predict(X_test_processed)

        ACCURACY_DIC[tokenization].append(accuracy_score(y_test, y_pred))
        VOCAB_DIC[tokenization].append(len(regressor.coef_[0]))

    print('i = ' + str(i))
    



# Encode accuracy results
filename = 'PED_Acc_n11_N100.csv'

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for tokenization in tokenizations:
        writer.writerow(ACCURACY_DIC[tokenization])


# Encode vocabulary results
filename = 'PED_Voc_n11_N100.csv'

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for tokenization in tokenizations:
        writer.writerow(VOCAB_DIC[tokenization])