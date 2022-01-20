import numpy as np

class Sample:

    def __init__(self, id, real_class, sentence, file_name):
        self.id = id
        self.real_class = real_class
        self.sentence = sentence
        self.file_name = file_name
        self.embedding = None
    
    def compute_representation(self, embedder, normalize=False):
        x = embedder.emb_sentence(self.sentence, normalize)
        self.embedding = np.mean(x, axis=0).reshape(1,-1)
    
    def print_data(self):
        print(f"{self.id}\t{self.real_class.name}\t{self.file_name}\t{self.sentence}")