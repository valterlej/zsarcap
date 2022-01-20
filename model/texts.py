import unicodedata
import re
import json
import numpy as np
from src.embedding import TransformerEmbedder


def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w, pad_punctuation=True, only_letters_and_punctuation=True):
    w = unicode_to_ascii(w.lower().strip())
    if pad_punctuation:
        w = re.sub(r"([?.!,¿])", r" \1 ", w) # creating a space between a word and the punctuation following it
        w = re.sub(r'[" "]+', " ", w)
    if only_letters_and_punctuation:
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = w.strip()  
    return w


def load_samples_from_files(sample_files):

    data = {}

    for sample_file in sample_files:
        
        predictions = json.loads(open(sample_file).read())["results"]
        for id, sample in enumerate(list(predictions.keys())):     
            file_name = sample
            sentence = predictions[sample][0]["sentence"].lower()
            try:
                d = data[file_name]
                d["sentences"].append(sentence)
            except:
                data[file_name] = {"sentences":[]}                
                data[file_name]["sentences"].append(sentence)

    return data