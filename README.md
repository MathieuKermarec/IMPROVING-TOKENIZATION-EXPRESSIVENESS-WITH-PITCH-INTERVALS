# IMPROVING TOKENIZATION EXPRESSIVENESS WITH PITCH INTERVALS

Repository for the article with the same name submitted in the Late-Breaking Demo session of ISMIR 2022.

- The Jupyter Notebook *ExploratoryWork.ipynb* contains a quick overview of the Miditok package, then nested evaluations and draft preliminary tests of some of the MIR tasks overviewed in the article.
- *toolbox.py* gathers many (including poorly named) utility functions useful for the notebook and the comparison algorithms.
- *PhraseEndDetection.py* and *ComposerClassification.py* are executive programs that tokenize a dataset through multiple strategies, then train and score a logistic regression model on two different MIR tasks.
- *DatasetPerplexity.py* is an attempt of unsupervised evaluation of those tokenization strategies.
- *SpatialPitchshift.py*, *Pitchshift.py* and *Pitchmute.py* are the three newly proposed tokenizations that are compared to the REMI and CPWord strategies. These codes are directly adapted from the *remi.py* program included in the Miditok package.

The authors are grateful to the Algomus and Magnet teams for fruitful discussions. This work is supported by a special interdisciplinary funding (AIT) from the CRIStAL laboratory and the Merlion PHC Music Language Processing N°48304SM funded by Campus France.
