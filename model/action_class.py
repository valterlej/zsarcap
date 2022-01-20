import numpy as np
from sklearn.neighbors import DistanceMetric

class ActionClass:

    def __init__(self, id, name, label, sentences):
        self.id = id
        self.name = name # name with spaces between words
        self.label = label # name without spaces or undescore
        self.sentences = sentences # a list containing the descriptive sentences
        self.sentences_representation = [] # a list containing the embedding representation for each sentence
        self.sentence_distances = [] # distance of a sentence from their label representation
        self.label_representation = None # an array representing the class label in embedding space
        self.document_representation = None # an array representing a class document representation (the concatenation of selected sentences)


    def compute_representations(self, embedder, normalize=False):
        # class label embedding
        x = embedder.emb_sentence(self.name, normalize)
        self.label_representation = np.mean(x, axis=0).reshape(1,-1)

        # sentences embedding
        for s in self.sentences:
            x = embedder.emb_sentence(s)
            self.sentences_representation.append(np.mean(x, axis=0).reshape(1,-1))   
        self.document_representation = embedder.emb_sentence(" ".join(self.sentences))

        dist = DistanceMetric.get_metric('euclidean')
        for sent_rep in self.sentences_representation:
            self.sentence_distances.append(dist.pairwise([self.label_representation[0], sent_rep[0]])[0][1])

        try:
            m = max(self.sentence_distances)
            self.sentence_distances = 1 - self.sentence_distances / (m + 0.1)
        except Exception as e:
            print(e)
            print(self.name)
            print(self.label)

    def print_id_name(self):
        print(f"{self.id}\t{self.name}")

    def print_data(self):
        print(f"{self.id}\t{self.name}")
        for i, s in enumerate(self.sentences):
            print(f"\t\t{s}\t{self.sentence_distances[i]}")
        