#!pip install contractions
import glob

import contractions
import nltk
import json
import collections
import numpy as np
from sentence_transformers import util
from tqdm import tqdm
from model.texts import preprocess_sentence
from src.embedding import TransformerEmbedder

nltk.download('punkt')

def proccess_paragraphs(lines, min_len):
    
    if not isinstance(lines, list):
        lines = [lines]

    paragraphs = []
    for l in lines:
        l = l[:-1].lower()
        if len(l) != 0:
            paragraphs.append(l)
    sentences = []
    for p in paragraphs:
        s = nltk.tokenize.sent_tokenize(p)
        for i in s:
            i = contractions.fix(i)
            words = i.split(" ")
            if len(words) >= min_len:
                sentences.append(preprocess_sentence(i, pad_punctuation=False, only_letters_and_punctuation=True))
    return sentences


def load_sentecens(file, min_len=10):
    lines = open(file,"r", encoding="cp1251", errors='ignore').readlines()
    return proccess_paragraphs(lines, min_len)    


def evaluate_similarity(sentences, class_embedding, embedder, max_sentences_per_file=10):   
    sentences_list = []
    embs = embedder.emb_sentence(sentences)
    cosine_scores = list(util.pytorch_cos_sim(class_embedding, embs).numpy()[0])
    sentences_list = [[x[0],x[1]] for x in zip(sentences, cosine_scores)]
    sentences_list = sorted(sentences_list, key=lambda item: -item[1])[:max_sentences_per_file]
    return [s[0] for s in sentences_list]


def load_class_sentences(dataset_dir, embedder, min_len=10, max_sentences_per_file=10, return_json=False):
    files = glob.glob(dataset_dir+"*.txt")
    class_sentences = {}
    for file in tqdm(files):        
        class_name = file.split("/")[-1].lower().replace("_"," ")[:-4]
        class_embedding = embedder.emb_sentence(class_name)
        sentences = load_sentecens(file, min_len)
        sentences = evaluate_similarity(sentences, class_embedding, embedder, max_sentences_per_file)
        if len(sentences) < max_sentences_per_file: ### showing if there are classes with less than max_sent
            print(f"{class_name} has only \t{len(sentences)} sentences.")
        class_sentences[class_name.replace(" ","_")] = sentences

    if return_json:
        return class_sentences
    else:
        sentences_list = []
        for cname in list(class_sentences.keys()):
            sentences = class_sentences[cname]
            for s in sentences:
                sentences_list.append([cname, s])
        return sentences_list