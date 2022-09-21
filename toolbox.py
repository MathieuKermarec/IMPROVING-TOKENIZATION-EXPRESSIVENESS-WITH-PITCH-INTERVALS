##################### COMPARING TOKENIZATION METHODS #########################

### AUXILIARY FUNCTIONS

import miditok
# Imports
from pathlib import Path
import os
from miditoolkit import MidiFile
import numpy as np

import sys

# These are the six tokenizations in miditok
from miditok import midi_like, structured, remi, cp_word, octuple_mono, mumidi, pitchshift, pitchmute, spatialpitchshift


# Returns a tokenizer with default parameters for a given tokenization method
def tokenizer_method(tokenization_method):
    tokenizers = [midi_like.MIDILike(), structured.Structured(), remi.REMI(), cp_word.CPWord(), octuple_mono.OctupleMono(), mumidi.MuMIDI(), pitchshift.PITCHSHIFT()]
    return tokenizers[tokenization_method]


# Returns the depth of a list of lists
def depth(lst):
    return isinstance(lst, list) and max(map(depth, lst or [0])) + 1


# Create a list of paths to our midi files
def create_midi_paths(dataset_path):
    from os import walk, path
    midi_paths = []
    for root, dirs, files in walk(path.normpath(dataset_path)):
        for file in files:
            if file.endswith(".mid"):
                midi_paths.append(path.join(root, file))
    return midi_paths


# A validation method to discard MIDIs we do not want during the main treatment
def midi_valid(midi) -> bool:
    if any(ts.numerator != 4 or ts.denominator != 4 for ts in midi.time_signature_changes):
        return False  # time signature different from 4/4
    if midi.max_tick < 10 * midi.ticks_per_beat:
        return False  # this MIDI is too short
    return True

# Transforms a sequence of compound word into a string of CP + number based on a lexicon.
# Updates the lexicon with the vocabulary from the token sequence

# input : tokens = [[[3, 1, 4, 93, 126, 224, 242, 252], 
#                    [3, 192, 4, 93, 126, 224, 242, 265], 
#                                  ... 
#                    [2, 191, 53, 113, 156, 224, 242, 252]]]
#
#         lexique = { [3, 1, 4, 93, 126, 224, 242, 252] : 'CP1',
#                     [3, 192, 4, 93, 126, 224, 242, 265] : 'CP2', 
#                                  ... 
#                     [2, 191, 53, 113, 156, 224, 242, 252] : 'CP87491' }
# output : cp_sequence = 'CP1 CP2 CP186 CP34 CP489 ... CP3678'
#
#          lexique_updated_with_tokens_vocabulary = ...

def tokens_to_CP(tokens, lexique):
    cp_sequence = ''
    for token in tokens:
        key = str(token)
        if key in lexique:
            cp_sequence += lexique[key] + ' '
        else:
            lexique[key] = 'CP'+str(len(lexique)+1)
            cp_sequence += lexique[key] + ' '
    return cp_sequence, lexique



### MAIN FUNCTIONS

def tokenize_dataset_single(dataset_path): # tokenization_method
    midi_paths = create_midi_paths(dataset_path)   # list of the paths to our files
    # tokenizer = tokenizer_method(tokenization_method)
    tokenizer = pitchshift.PITCHSHIFT()

    dataset_vocabulary = {} # = { token : number of occurrencies in the sequences from the dataset }
                            # = { 1 : 45637, 2 : 34087, ... , 208 : 13876 }
    
    # token sequence data initialisation
    average_token_length = 0
    min_token_length = 10^9
    max_token_length = 0
    
    corpus = [] # if we want to create a corpus of strings for NLP
                # = ['1 23 45 205 23 ... 206', '1 23 48 245 23 ... 207', ...]
        
    compteur = 0 # to keep track of our tokenization
    
    for m, midi_path in enumerate(midi_paths):
        
        # initialise
        try:
            midi = MidiFile(midi_path)
        except FileNotFoundError:
            print(f'File not found: {midi_path}')
            continue
        except Exception as _:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
            continue

        midi = MidiFile(midi_path)
        
        # Passing the MIDI to validation tests
        if not midi_valid(midi):
            continue
            
        tokens = tokenizer.midi_to_tokens(midi)
    
        # Updating data
        if depth(tokens)==2: # verifies that our token is in the format we want      
            
            # Token information
            token_sequence_length = len(tokens[0])
            average_token_length += token_sequence_length
            min_token_length = min(min_token_length, token_sequence_length)
            max_token_length = max(max_token_length, token_sequence_length)
                       
            # Corpus for NLP
            corpus_sequence = ''.join(str(nb) + ' ' for nb in tokens[0])
            corpus.append(corpus_sequence)
        
            # Dataset vocabulary
            vocabulary, counts = np.unique(np.array(tokens[0]), axis=0, return_counts=True)
            for i in range(len(vocabulary)):
                token = str(vocabulary[i])
                if token in dataset_vocabulary:
                    dataset_vocabulary[token] += counts[i]
                else:
                    dataset_vocabulary[token] = counts[i]
            
            # to keep track of our tokenization
            compteur += 1
            if compteur%100 == 0:
                print(str(compteur) + ' files treated ...')
    
    # Show relevant data
    print('All ' + str(compteur) + ' files treated !')
    print('Average token sequence length =', round(average_token_length/len(midi_paths)))  
    print('Minimum token sequence length =', min_token_length)
    print('Maximum token sequence length =', max_token_length)
    print('Dataset vocabulary size =', len(dataset_vocabulary))
    print('Dataset total number of tokens =', sum(len(sequence.split()) for sequence in corpus))
    
    return [dataset_vocabulary, average_token_length, min_token_length, max_token_length, corpus]




def tokenize_dataset_grouped(dataset_path, tokenizer):
    midi_paths = create_midi_paths(dataset_path)   # list of the paths to our files
    
    dataset_vocabulary = {} # = { token : number of occurrencies in the sequences from the dataset }
                            # = { [1, 23, 45, 206] : 1230, [1, 24, 45, 206] : 82, ... }
    
    # token sequence data initialisation
    average_token_length = 0
    min_token_length = 10^9
    max_token_length = 0
    
    # if we want to create a corpus of strings for NLP
    corpus = [] # = ['CP1 CP23 CP45 CP205 ... CP206', 'CP1 CP23 CP48 CP245 ... CP207', ...]
    lexique = {} # = { token : corresponding CP Word }
                 # = { [1, 23, 45, 206] : 'CP1', [1, 24, 45, 206] : 'CP2', ... , [1, 76, 83, 108] : 'CP8931'}
        
    compteur = 0 # to keep track of the number of files tokenized
    
    for m, midi_path in enumerate(midi_paths):
        
        # initialise
        try:
            midi = MidiFile(midi_path)
        except FileNotFoundError:
            print(f'File not found: {midi_path}')
            continue
        except Exception as _:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
            continue
            
        midi = MidiFile(midi_path)
        
        # Passing the MIDI to validation tests
        if not midi_valid(midi):
            continue
        
        tokens = tokenizer.midi_to_tokens(midi)
    
        # Updating data
        if depth(tokens)==3: # verifies that our token is in the format we want      
        
            # Token information
            token_sequence_length = len(tokens[0])
            average_token_length += token_sequence_length
            min_token_length = min(min_token_length, token_sequence_length)
            max_token_length = max(max_token_length, token_sequence_length)
                       
            # Corpus for NLP
            CP_sequence, lexique = tokens_to_CP(tokens[0], lexique)
            corpus.append(CP_sequence)
        
            # Dataset vocabulary
            vocabulary, counts = np.unique(np.array(tokens[0]), axis=0, return_counts=True)
            for i in range(len(vocabulary)):
                token = str(vocabulary[i])
                if token in dataset_vocabulary:
                    dataset_vocabulary[token] += counts[i]
                else:
                    dataset_vocabulary[token] = counts[i]
            
            # to keep track of our tokenization
            compteur += 1
            if compteur%100 == 0:
                print(str(compteur) + ' files treated ...')
    
    # Show relevant data
    print('All ' + str(compteur) + ' files treated !')
    print('Average token sequence length =', round(average_token_length/len(midi_paths)))  
    print('Minimum token sequence length =', min_token_length)
    print('Maximum token sequence length =', max_token_length)
    print('Dataset vocabulary size =', len(dataset_vocabulary))
    print('Dataset total number of tokens =', sum(len(sequence.split()) for sequence in corpus))
    
    return [dataset_vocabulary, average_token_length, min_token_length, max_token_length, corpus, lexique]


    ##################### NLP CLASSIFICATION #########################

### AUXILIARY FUNCTIONS

from math import floor
import miditoolkit
import random
import mido as mido

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns

# Creates a list y = [['Bach'],['Bach'], ... , ['Schubert']] 
# from a dataset of MIDI files whose names start by the author followed by a comma
# for example : Bach, Johann Sebastian, 15 Sinfonias, BWV 787-801, ntLGHRX5XOE.mid

def create_y_from_dataset(dataset_path):
    y = []
    for root, dirs, files in os.walk(os.path.normpath(dataset_path)):
        for file in files:
            if file.endswith(".mid"):
                y.append(file.split(",")[:1][0])
    return y


# Returns the first tempo data of a mido MidiFile object
def get_tempo(mid):
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
            else:
                # Default tempo.
                return 500000


# Returns the closest tick corresponding to a beat of a MidiFile object
def closest_beat_from_tick(tick, ticks_per_beat):
    return ticks_per_beat * (floor(tick/ticks_per_beat))


# Dumps a midi file 'extrait.mid' with a duration of duree_extrait_secondes seconds
def extract_from_midi(midi_path, duree_extrait_secondes, n):
    mid = mido.MidiFile(midi_path)
    mid_as_miditoolkit = miditoolkit.MidiFile(midi_path)
    tempo_midi = get_tempo(mid) # sets the tempo to the first tempo token in the MIDI ...
    duree_extrait_ticks = (mid.ticks_per_beat * duree_extrait_secondes / (tempo_midi / 10**6))

    duree_midi_ticks = (mid.ticks_per_beat * mid.length / (tempo_midi / 10**6))

    if mid.length - duree_extrait_secondes >= 0:
        random_tick = random.randrange(0, int(duree_midi_ticks - duree_extrait_ticks))
        start_tick = closest_beat_from_tick(random_tick, mid.ticks_per_beat)
        end_tick = start_tick + duree_extrait_ticks
        filename = 'extrait' + str(n) + '.mid'
        mid_as_miditoolkit.dump(filename = filename, segment = [start_tick, end_tick])
    else:
        filename = 'extrait' + str(n) + '.mid'
        mid_as_miditoolkit.dump(filename = filename) # MIDI entier
    return


# Returns [duration in seconds, number of notes] of a midi file
def duration_and_notes(midi_path):
    mid = mido.MidiFile(midi_path)
    nb_of_notes = 0
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on':
                nb_of_notes += 1
    return [mid.length, nb_of_notes]

# Prints te cumulated duration of the midi files of a dataset, as well as
# average duration and cumulated number of notes.
def corpus_duration_and_notes(corpus_path):
    midi_paths = create_midi_paths(corpus_path)
    total_duration = 0
    total_nb_of_notes = 0
    for m, midi_path in enumerate(midi_paths):
        data = duration_and_notes(midi_path)
        total_duration += data[0]
        total_nb_of_notes += data[1]
    print('Cumulated duration of files in the dataset : ' + str(total_duration) + ' seconds.')
    print('Average duration of files in the dataset : ' + str(round(total_duration/len(midi_paths))) + ' seconds.')
    print('Number of notes in the dataset : ' + str(total_nb_of_notes) + ' notes.')
    return [total_duration, len(midi_paths), total_nb_of_notes]




# Creates a list res = [['Classed Author'], ['Classed Author'], ['Other'], ... , ['Other']]
# from a list y of authors and a string classed_author of the author's name we want to classiffy
def create_y_binary(y, classed_author):
    res = []
    for author in y:
        if author == classed_author:
            res.append([classed_author])
        else :
            res.append(['Other'])
    return res


def create_midi_paths_author(dataset_path, author):
    from os import walk, path
    midi_paths = []
    for root, dirs, files in walk(path.normpath(dataset_path)):
        for file in files:
            if file.endswith(".mid"):
                if file.startswith(author):
                    midi_paths.append(path.join(root, file))
    return midi_paths


    ###################### Creating Uniform Datasets of token sequences for NLP classification #####################

# Creates a dataset of (number_of_authors * number_of_files_per_author) tokenized midi files given :
#       - the path to the dataset containing the authors' midi files
#       - the list of authors we want in the dataset
#       - the duration of the midis extracted from those authors' midis files
#       - the quantity of midis extracted for each author
#       - the tokenization method

from toolbox import tokens_to_CP


def create_sampled_dataset_single(dataset_path, authors_list, duration_midis, quantity_midis, tokenization_method):
    corpus = []

    tokenizer = tokenizer_method(tokenization_method)

    for author in authors_list:
        midi_paths = create_midi_paths(dataset_path + "/" + author)   # list of the paths to the author files
        for i in range(1, quantity_midis + 1):
            midi_path = random.choice(midi_paths) ### 
            extract_from_midi(midi_path, duration_midis) # creates 'extrait.mid'
            midi = miditoolkit.MidiFile('extrait.mid')
            tokens = tokenizer.midi_to_tokens(midi)
            corpus_sequence = ''.join(str(nb) + ' ' for nb in tokens[0])
            corpus.append(corpus_sequence)
            os.remove('extrait.mid')
            if i%100 == 0:
                print (str(i) + ' files treated ...')
        print(author + "'s files have been tokenized !")

    return corpus

def create_sampled_dataset_grouped(dataset_path, authors_list, duration_midis, quantity_midis, tokenization_method):
    corpus = []
    lexique = {}

    tokenizer = tokenizer_method(tokenization_method)

    for author in authors_list:
        midi_paths = create_midi_paths(dataset_path + "/" + author)   # list of the paths to the author files
        for i in range(1, quantity_midis + 1):
            midi_path = random.choice(midi_paths) ### 
            extract_from_midi(midi_path, duration_midis) # creates 'extrait.mid'
            midi = miditoolkit.MidiFile('extrait.mid')
            tokens = tokenizer.midi_to_tokens(midi)
            CP_sequence, lexique = tokens_to_CP(tokens[0], lexique)
            corpus.append(CP_sequence)
            os.remove('extrait.mid')
            if i%100 == 0:
                print (str(i) + ' files treated ...')
        print(author + "'s files have been tokenized !")

    return corpus



# function to return key for any value
def get_key(dict, val):
    for key, value in dict.items():
         if val == value:
             return key
    return "key doesn't exist"



from sklearn.model_selection import ShuffleSplit

def create_sampled_dataset(dataset_path, authors_list, duration_midis, quantity_midis, test_size, random_state, tokenizations):
    
    CORPUS_DIC = {
        'REMI': [[], [], [], []], # X_train, X_test, y_train, y_test
        'CPWord': [[], [], [], []],
        'PitchShift': [[], [], [], []],
        'PitchMute': [[], [], [], []],
        'SpatialPS': [[], [], [], []]
    }

    VOCAB_DIC = {
        'REMI': {}, 
        'CPWord': {},
        'PitchShift': {},
        'PitchMute': {},
        'SpatialPS': {}
    }

    lexique = {}

    TOKENIZER_DIC = {
        'REMI': remi.REMI(),
        'CPWord': cp_word.CPWord(),
        'PitchShift': pitchshift.PITCHSHIFT(),
        'PitchMute': pitchmute.PITCHMUTE(),
        'SpatialPS': spatialpitchshift.SPATIALPITCHSHIFT()
    }

    rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_paths = []
    test_paths = []

    for author in authors_list:
        midi_paths = create_midi_paths(dataset_path + "/" + author)   # list of the paths to the author files
        
        train_dic_author = {}
        test_dic_author = {}

        for train_index, test_index in rs.split(midi_paths):    # create dictionnary linking file names and indexes
            for indice in train_index :
                train_dic_author[midi_paths[indice]] = indice
            for indice in test_index :
                test_dic_author[midi_paths[indice]] = indice
        
        # create train corpus

        for i in range(1, int(quantity_midis*(1 - test_size)) + 1):

            midi_path = get_key(train_dic_author, random.choice(train_index)) ###
            train_paths.append(midi_path) # save the path of the selected file

            extract_from_midi(midi_path, duration_midis) # creates 'extrait.mid'
            midi = miditoolkit.MidiFile('extrait.mid')

            for tokenization in tokenizations: # ['REMI', 'CPWord', 'PitchShift', 'PitchMute']
                tokenizer = TOKENIZER_DIC[tokenization]
                token_sequence = tokenizer.midi_to_tokens(midi)
                if token_sequence != []:
                    if tokenization == 'CPWord':
                        CP_sequence, lexique = tokens_to_CP(token_sequence[0], lexique)
                        CORPUS_DIC[tokenization][0].append(CP_sequence) # update X_train
                        CORPUS_DIC[tokenization][2].append([author]) # update y_train
                        VOCAB_DIC[tokenization] = update_vocab_grouped(VOCAB_DIC[tokenization], tokenizer, token_sequence)
                    else:
                        corpus_sequence = ''.join(str(nb) + ' ' for nb in token_sequence[0])
                        CORPUS_DIC[tokenization][0].append(corpus_sequence) # update X_train
                        CORPUS_DIC[tokenization][2].append([author]) # update y_train
                        VOCAB_DIC[tokenization] = update_vocab_single(VOCAB_DIC[tokenization], tokenizer, token_sequence)

            os.remove('extrait.mid')

            if i%100 == 0:
                print (str(i) + ' files treated ...')
        print(author + "'s train files have been tokenized !")

        # create test corpus
        
        for i in range(1, int(quantity_midis*test_size) + 1):

            midi_path = get_key(test_dic_author, random.choice(test_index)) ###
            test_paths.append(midi_path) # save the path of the selected file

            extract_from_midi(midi_path, duration_midis) # creates 'extrait.mid'
            midi = miditoolkit.MidiFile('extrait.mid')

            for tokenization in tokenizations: # ['REMI', 'CPWord', 'PitchShift', 'PitchMute']
                tokenizer = TOKENIZER_DIC[tokenization]
                token_sequence = tokenizer.midi_to_tokens(midi)
                if token_sequence != []:
                    if tokenization == 'CPWord':
                        CP_sequence, lexique = tokens_to_CP(token_sequence[0], lexique)
                        CORPUS_DIC[tokenization][1].append(CP_sequence) # update X_train
                        CORPUS_DIC[tokenization][3].append([author]) # update y_train
                        VOCAB_DIC[tokenization] = update_vocab_grouped(VOCAB_DIC[tokenization], tokenizer, token_sequence)
                    else:
                        corpus_sequence = ''.join(str(nb) + ' ' for nb in token_sequence[0])
                        CORPUS_DIC[tokenization][1].append(corpus_sequence) # update X_train
                        CORPUS_DIC[tokenization][3].append([author]) # update y_train
                        VOCAB_DIC[tokenization] = update_vocab_single(VOCAB_DIC[tokenization], tokenizer, token_sequence)

            os.remove('extrait.mid')

            if i%100 == 0:
                print (str(i) + ' files treated ...')
        print(author + "'s test files have been tokenized !")

    data_shuffle = [train_paths, test_paths]

    corpuses = []
    vocabularies = []
    for tokenization in tokenizations:
        corpuses.append(CORPUS_DIC[tokenization])
        if tokenization == 'CPWord':
            vocabularies.append([VOCAB_DIC[tokenization], lexique])
        else:
            vocabularies.append(VOCAB_DIC[tokenization])

    return corpuses, data_shuffle, vocabularies



def create_sampled_dataset_fast(dataset_path, authors_list, duration_midis, quantity_midis, test_size, random_state, tokenizations, data_augmented, ngram_range):
    
    CORPUS_DIC = {
        'REMI': [[], [], [], []], # X_train, X_test, y_train, y_test
        'CPWord': [[], [], [], []],
        'PitchShift': [[], [], [], []],
        'PitchMute': [[], [], [], []],
        'SpatialPS': [[], [], [], []]
    }

    lexique = {}

    TOKENIZER_DIC = {
        'REMI': remi.REMI(),
        'CPWord': cp_word.CPWord(),
        'PitchShift': pitchshift.PITCHSHIFT(),
        'PitchMute': pitchmute.PITCHMUTE(),
        'SpatialPS': spatialpitchshift.SPATIALPITCHSHIFT()
    }

    rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_paths = []
    test_paths = []

    for author in authors_list:
        print(dataset_path + "/" + author)
        midi_paths = create_midi_paths(dataset_path + "/" + author)   # list of the paths to the author files
        
        train_dic_author = {}
        test_dic_author = {}
        print(len(midi_paths))
        for train_index, test_index in rs.split(midi_paths):    # create dictionnary linking file names and indexes
            for indice in train_index :
                train_dic_author[midi_paths[indice]] = indice
            for indice in test_index :
                test_dic_author[midi_paths[indice]] = indice
        
        # create train corpus

        for i in range(1, int(quantity_midis*(1 - test_size)) + 1):

            midi_path = get_key(train_dic_author, random.choice(train_index)) ###
            train_paths.append(midi_path) # save the path of the selected file

            extract_from_midi(midi_path, duration_midis, 1) # creates 'extrait.mid'
            midi = miditoolkit.MidiFile('extrait.mid')

            for tokenization in tokenizations: # ['REMI', 'CPWord', 'PitchShift', 'PitchMute']
                tokenizer = TOKENIZER_DIC[tokenization]
                token_sequence = tokenizer.midi_to_tokens(midi)
                if token_sequence != []:
                    if tokenization == 'CPWord':
                        CP_sequence, lexique = tokens_to_CP(token_sequence[0], lexique)
                        CORPUS_DIC[tokenization][0].append(CP_sequence) # update X_train
                        CORPUS_DIC[tokenization][2].append([author]) # update y_train
                    else:
                        corpus_sequence = ''.join(str(nb) + ' ' for nb in token_sequence[0])
                        CORPUS_DIC[tokenization][0].append(corpus_sequence) # update X_train
                        CORPUS_DIC[tokenization][2].append([author]) # update y_train

            if data_augmented:
                transpose_midi('extrait.mid')
                for i in range(1, 12):
                    midi = miditoolkit.MidiFile('extrait_+' + str(i) + '.mid')
                    for tokenization in tokenizations:
                        tokenizer = TOKENIZER_DIC[tokenization]
                        token_sequence = tokenizer.midi_to_tokens(midi)
                        if token_sequence != []:
                            if tokenization == 'REMI':
                                corpus_sequence = ''.join(str(nb) + ' ' for nb in token_sequence[0])
                                CORPUS_DIC[tokenization][0].append(corpus_sequence) # update X_train
                                CORPUS_DIC[tokenization][2].append([author]) # update y_train
                            if tokenization == 'CPWord':
                                CP_sequence, lexique = tokens_to_CP(token_sequence[0], lexique)
                                CORPUS_DIC[tokenization][0].append(CP_sequence) # update X_train
                                CORPUS_DIC[tokenization][2].append([author]) # update y_train
                    os.remove('extrait_+' + str(i) + '.mid')

            os.remove('extrait.mid')

            if i%10 == 0:
                print (str(i) + ' files treated ...')
        print(author + "'s train files have been tokenized !")

        # create test corpus
        
        for i in range(1, int(quantity_midis*test_size) + 1):

            midi_path = get_key(test_dic_author, random.choice(test_index)) ###
            test_paths.append(midi_path) # save the path of the selected file

            extract_from_midi(midi_path, duration_midis, 1) # creates 'extrait.mid'
            midi = miditoolkit.MidiFile('extrait.mid')

            for tokenization in tokenizations: # ['REMI', 'CPWord', 'PitchShift', 'PitchMute']
                tokenizer = TOKENIZER_DIC[tokenization]
                token_sequence = tokenizer.midi_to_tokens(midi)
                if token_sequence != []:
                    if tokenization == 'CPWord':
                        CP_sequence, lexique = tokens_to_CP(token_sequence[0], lexique)
                        CORPUS_DIC[tokenization][1].append(CP_sequence) # update X_train
                        CORPUS_DIC[tokenization][3].append([author]) # update y_train
                    else:
                        corpus_sequence = ''.join(str(nb) + ' ' for nb in token_sequence[0])
                        CORPUS_DIC[tokenization][1].append(corpus_sequence) # update X_train
                        CORPUS_DIC[tokenization][3].append([author]) # update y_train
                
            os.remove('extrait.mid')

            if i%10 == 0:
                print (str(i) + ' files treated ...')
        print(author + "'s test files have been tokenized !")
    
    accuracy_scores = {}
    size_vocabs = {}
    for tokenization in tokenizations:
        CORPUS = CORPUS_DIC[tokenization]
        X_train, X_test, y_train, y_test = CORPUS[0], CORPUS[1], CORPUS[2], CORPUS[3]
        y_true = y_test # Useless but still

        # Initiate regressor and vectorizer
        regressor = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter=10000) # n_grams
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)

        # Process data
        X_train_processed = vectorizer.fit_transform(X_train)
        X_test_processed = vectorizer.transform(X_test)
        regressor = regressor.fit(X_train_processed, y_train)
        y_pred = regressor.predict(X_test_processed)

        accuracy_scores[tokenization] = accuracy_score(y_test, y_pred)
        size_vocabs[tokenization] = len(regressor.coef_[0])

    return accuracy_scores, size_vocabs




def get_perplexity(dataset_path, test_size, random_state, tokenizations):

    CORPUS_DIC = {
        'REMI': [[], []], # train, test
        'CPWord': [[], []],
        'PitchShift': [[], []],
        'PitchMute': [[], []],
        'SpatialPS': [[], []]
    }

    lexique = {}

    TOKENIZER_DIC = {
        'REMI': remi.REMI(),
        'CPWord': cp_word.CPWord(),
        'PitchShift': pitchshift.PITCHSHIFT(),
        'PitchMute': pitchmute.PITCHMUTE(),
        'SpatialPS': spatialpitchshift.SPATIALPITCHSHIFT()
    }

    rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_paths = []
    test_paths = []

    midi_paths = []
    authors_list = ['Bach', 'Beethoven', 'Chopin', 'Mozart', 'Liszt', 'Schubert']

    for author in authors_list:
        midi_paths_author = create_midi_paths(dataset_path + "/" + author)   # list of the paths to the author files
        midi_paths.extend(midi_paths_author)

    for train_index, test_index in rs.split(midi_paths):
        view_count = 0
        for indice in train_index:
            midi_path = midi_paths[indice]
            midi = miditoolkit.MidiFile(midi_path)
            for tokenization in tokenizations: # ['REMI', 'CPWord', 'PitchShift', 'PitchMute']
                tokenizer = TOKENIZER_DIC[tokenization]
                token_sequence = tokenizer.midi_to_tokens(midi)
                if token_sequence != []:
                    if tokenization == 'CPWord':
                        CP_sequence, lexique = tokens_to_CP(token_sequence[0], lexique)
                        corpus_sequence = CP_sequence.split()
                        CORPUS_DIC[tokenization][0].append(corpus_sequence) # update train
                    else:
                        corpus_sequence = [str(nb) for nb in token_sequence[0]]
                        CORPUS_DIC[tokenization][0].append(corpus_sequence) # update train
            view_count += 1
            if view_count%100 == 0:
                print(str(view_count) + ' files treated ...')
        print('All train files tokenized !')

        view_count = 0
        for indice in test_index:
            midi_path = midi_paths[indice]
            midi = miditoolkit.MidiFile(midi_path)
            for tokenization in tokenizations: # ['REMI', 'CPWord', 'PitchShift', 'PitchMute']
                tokenizer = TOKENIZER_DIC[tokenization]
                token_sequence = tokenizer.midi_to_tokens(midi)
                if token_sequence != []:
                    if tokenization == 'CPWord':
                        CP_sequence, lexique = tokens_to_CP(token_sequence[0], lexique)
                        corpus_sequence = CP_sequence.split()
                        CORPUS_DIC[tokenization][1].append(corpus_sequence) # update test
                    else:
                        corpus_sequence = [str(nb) for nb in token_sequence[0]]
                        CORPUS_DIC[tokenization][1].append(corpus_sequence) # update test
            view_count += 1
            if view_count%100 == 0:
                print(str(view_count) + ' files treated ...')
        print('All test files tokenized !')

    return CORPUS_DIC


from nltk.lm import MLE, Laplace, Lidstone, StupidBackoff
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

def LogLikelihood(test_sequence, lm, N, smoothing, alpha):
    res = 1

    pad_both_ends(test_sequence, n = N)

    #if smoothing == "StupidBackOff":
    #    lm = StupidBackoff(alpha=alpha, order=N)
    #if smoothing == "Lidstone":
    #    lm = Lidstone(gamma=alpha, order=N)
    #if smoothing == "Laplace":
    #    lm = Lidstone(gamma=1, order=N)

    if N == 1:
        for ngram in test_sequence:
            res += lm.logscore(ngram)

    elif N == 2:
        for i in range(0, len(test_sequence) - 1):
            res += lm.logscore(test_sequence[::-1][i], test_sequence[::-1][i+1].split())
        res += lm.logscore(test_sequence[0])

    elif N == 3:
        for i in range(0, len(test_sequence) - 2):
            res += lm.logscore(test_sequence[::-1][i], str(test_sequence[::-1][i+2] + " " + test_sequence[::-1][i+1]).split())
        res += lm.logscore(test_sequence[1], test_sequence[0].split())
        res += lm.logscore(test_sequence[0])
    
    return res



def END_sequence_split(sequence, phrase_length, nb_phrases):
    phrases, END_presence_list = [], []

    for i in range(nb_phrases):
        bar_indicator = sequence[sequence.index('END') + 1]
        bar_indexes = [i for i, x in enumerate(sequence) if x == bar_indicator]
        first_bar_index = random.randrange(0, len(bar_indexes) - phrase_length)
        phrase = sequence[bar_indexes[first_bar_index]:bar_indexes[first_bar_index + phrase_length]]

        END_presence_list.append(('END' in phrase))

        for token in phrase:
            if token == 'END':
                phrase = list(filter(lambda x: x != 'END', phrase))

        phrases.append(phrase)

    return phrases, END_presence_list



import mido
import pretty_midi

# Saves 11 transposed midis from a midi file

def transpose_midi(file_path):
    for i in range(1, 12):
        midi_transposed = pretty_midi.PrettyMIDI(file_path)
        for instrument in midi_transposed.instruments:
            if not instrument.is_drum:
                for note in instrument.notes: # the data is : note start, end, pitch, velocity and the name of the instrument.
                    note.pitch += i
        midi_transposed.write(file_path[:(len(file_path) - 4)] + '_+' + str(i) + '.mid')
    return

# Updates vocabulary of a tokenization

def update_vocab_single(vocab, tokenizer, token_sequence):
    for token in token_sequence[0]:
        vocab[str(token)] = str(tokenizer.tokens_to_events([token]))
    return vocab


def update_vocab_grouped(vocab, tokenizer, token_sequence):
    for grp_token in token_sequence[0]:
        vocab[str(grp_token)] = str(tokenizer.tokens_to_events(grp_token))
    return vocab




    ##################### BERTWordPieceTokenizer #########################

from mangoes.modeling import BERTWordPieceTokenizer

### AUXILIARY

# Creates a file of .txt each containing a sequence

def corpus_to_file(corpus, text_corpus_path):
    file_number = 0
    for token_sequence in corpus:
        text_file = open(text_corpus_path+str(file_number)+".txt", "w")
        text_file.write(token_sequence)
        text_file.close()
        file_number += 1
    return
        
# Create a list of paths to our text files
def create_text_paths(text_corpus_path):
    text_paths = []
    for root, dirs, files in os.walk(os.path.normpath(text_corpus_path)):
        for file in files:
            if file.endswith(".txt"):
                text_paths.append(os.path.join(root, file))
    return text_paths
