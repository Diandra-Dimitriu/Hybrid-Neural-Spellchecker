import nltk
import re
import json
from nltk.corpus import gutenberg


nltk.download('gutenberg')
nltk.download('punkt')


raw_text = gutenberg.raw('carroll-alice.txt')
clean_text = re.sub(r"[^a-z\s']", '', raw_text.lower())
corpus_words = clean_text.split()

with open('alice_corpus.json', 'w') as file:
    json.dump(corpus_words, file)

print("Corpus generated and saved successfully!")