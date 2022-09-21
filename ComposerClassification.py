import toolbox
from toolbox import create_sampled_dataset_fast, create_midi_paths
import csv

from optparse import OptionParser

# Options from the parser
parser = OptionParser()
parser.add_option("-N", type="int", dest="N", help="quantity of midis per author")
parser.add_option("-D", type="int", dest="D", help="duration of the samples")
parser.add_option("-A", type="str", dest="author1", help="first author")
parser.add_option("-B", type="str", dest="author2", help="second author")
(options, args) = parser.parse_args()

duration_midis = options.D # Duration (in seconds) of the samples we extract from midis
quantity_midis = options.N # Number of samples we want per author
authors_list = [options.author1, options.author2] # Pair of authors to classify

# Other options
dataset_path = './CustomCorpus'
test_size = 0.20
nb_loop = 10
tokenizations = ['REMI', 'CPWord', 'PitchShift', 'PitchMute', 'SpatialPS']

data_augmented = False
ngram_range = (1,1)

# Accuracy and Vocabulary dictionnaries, relative to each tokenization
ACCURACY_DIC = {}
VOCAB_DIC = {}
for tokenization in tokenizations:
    ACCURACY_DIC[tokenization] = [tokenization]
    VOCAB_DIC[tokenization] = [tokenization]

# Loop
for i in range(nb_loop):
    random_state = i+1

    accuracy_scores, size_vocabs = create_sampled_dataset_fast(dataset_path, authors_list, duration_midis, quantity_midis, test_size, random_state, tokenizations, data_augmented)
    
    for tokenization in tokenizations:
        ACCURACY_DIC[tokenization].append(accuracy_scores[tokenization])
        VOCAB_DIC[tokenization].append(size_vocabs[tokenization])


# Encode accuracy results
filename = 'Acc_N' + str(quantity_midis) + '_D' + str(duration_midis) + '_' + options.author1 + options.author2 + '.csv'

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for tokenization in tokenizations:
        writer.writerow(ACCURACY_DIC[tokenization])


# Encode vocabulary results
filename = 'Voc_N' + str(quantity_midis) + '_D' + str(duration_midis) + '_' + options.author1 + options.author2 + '.csv'

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for tokenization in tokenizations:
        writer.writerow(VOCAB_DIC[tokenization])

# rcparam matplotlib