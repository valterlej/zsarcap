import unicodedata
import re

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

def process_label(label):
    return label.lower().replace(" ","").replace("_","").replace("(","").replace(")","").replace("'","")