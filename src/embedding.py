import json
import spacy
import torch
import sent2vec
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer, util

lookup_replace_table = {"skijet":"jet ski"}

class GloveEmbedder:

    def __init__(self):
        self.nlp = spacy.load('en')
        self.n_vocab, self.vocab_dim = self.nlp.vocab.vectors.shape
        self.emb = nn.Embedding(self.n_vocab, self.vocab_dim)
        self.emb.weight.data.copy_(torch.from_numpy(self.nlp.vocab.vectors.data))
        self.key2row = self.nlp.vocab.vectors.key2row                
    
    def emb_sentence(self, sentences, normalize=False, return_mean=False):

        s_embs = []

        for s in sentences:
            for w in list(lookup_replace_table.keys()):
                s = s.replace(w,lookup_replace_table[w])

            s = s.strip().split()
            print(s)
            embs = []
            for v in s:            
                vocab_id = self.nlp.vocab.strings[v]
                print(vocab_id)
                row = self.key2row.get(vocab_id, None)
                if row is None:
                    print(f"\n{v}\n")
                    continue
                else:
                    vocab_row = torch.tensor(row, dtype=torch.long)
                    embed_vec = self.emb(vocab_row)
                    embs.append(embed_vec.detach().numpy().reshape(1,-1))
            
            if return_mean:
                s_embs.append(np.mean(np.concatenate(embs), axis=0).reshape(1,-1))
            else:
                s_embs.append(np.concatenate(embs))

        return np.concatenate(s_embs)

class Sent2vecEmbedder:

    def __init__(self, model_path='./data/wiki_bigrams.bin'):

        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(model_path, inference_mode=True)        
    
    def emb_sentence(self, sentences, normalize=False):
        
        embs = []
        for s in sentences:    
            x = self.model.embed_sentence(s.lower())
            if normalize:
                x = x / np.linalg.norm(x, axis=1, ord=1) + 1e-8
            embs.append(x)
        return np.concatenate(embs)

class TransformerEmbedder:

    def __init__(self, model_name="paraphrase-distilroberta-base-v2"):            
        self.model = SentenceTransformer(model_name)      
    
    def emb_sentence(self, sentences, normalize=False):
        x = self.model.encode(sentences)
        if not isinstance(sentences, list):            
            return x.reshape(1,-1)
        else:
            return x