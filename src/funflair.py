from flair.data import Sentence
from flair.models import MultiTagger
from flair.tokenization import SciSpacyTokenizer
import pandas as pd


test = pd.read_csv("datasets/cometa/splits/stratified_general/test.csv", sep = "\t")


# make a sentence and tokenize with SciSpaCy
sentence = Sentence("Behavioral abnormalities in the Fmr1 KO2 Mouse Model of Fragile X Syndrome", use_tokenizer=SciSpacyTokenizer())

# load biomedical tagger
tagger = MultiTagger.load("hunflair")

# tag sentence
tagger.predict(sentence)