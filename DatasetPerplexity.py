import toolbox
import csv
import pickle

import sys

from toolbox import get_perplexity, LogLikelihood

from nltk.lm import MLE, Laplace, Lidstone, StupidBackoff
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.metrics import BigramAssocMeasures # likelihood_ratio 
from optparse import OptionParser

# Options from the parser
parser = OptionParser()
parser.add_option("-n", type="int", dest="n", help="value of n in n-grams")
(options, args) = parser.parse_args()

n = options.n

# Other options
dataset_path = r'/home/mathieu/Desktop/TFE/test/CustomCorpus'
test_size = 0.20
# nb_loop = 10
tokenizations = ['REMI', 'CPWord', 'PitchShift', 'SpatialPS', 'PitchMute'] 
smoothings = ["Laplace"]

random_state = 300

CORPUS_DIC = get_perplexity(dataset_path, test_size, random_state, tokenizations)

#pickle.dump(CORPUS_DIC, open("Perplexity_CORPUS_DIC_rs" + str(random_state) + ".p", 'wb'))
#CORPUS_DIC = pickle.load(open("Perplexity_CORPUS_DIC_rs300.p", "rb"))

for i in range(1,n+1):
    logscores = {}
    for tokenization in tokenizations:
        for smoothing in smoothings:
            logscore = 0

            train, vocab_train = padded_everygram_pipeline(i, CORPUS_DIC[tokenization][0])

            lm = MLE(i)
            lm = Lidstone(gamma=1, order=i)
            lm.fit(train, vocab_train)

            for sequence in CORPUS_DIC[tokenization][1]:
                print(len(sequence))
                logscore += LogLikelihood(test_sequence = sequence, lm=lm, N=i, smoothing=smoothing, alpha=0.4)
                print(logscore)
                
                #test_sequence.extend(pad_both_ends(sequence, i))

            logscores[tokenization + "_" + smoothing] = logscore/len(CORPUS_DIC[tokenization][1])
            print(tokenization + "_" + smoothing, logscores[tokenization + "_" + smoothing])

    # Encode perplexity results
    filename = 'Logscores_n' + str(i) + '.csv'

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for tokenization in tokenizations:
            writer.writerow([logscores[tokenization + "_" + smoothing]])


