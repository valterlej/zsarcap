from model.action_class import ActionClass
from model.sample import Sample
from model.texts import preprocess_sentence, load_samples_from_files
from src.sentences_pre_processing import load_class_sentences, proccess_paragraphs
import json
import random
import numpy as np
from src.texts import process_label
from tqdm import tqdm

class Dataset:

    def __init__(self, name, file_classes, file_zsl_splits, dataset_info, min_class_sentence_len=15, max_class_sentences=10, 
                 file_samples = [], preprocess_embedder = None, zsar_embedder= None, load_samples=True, normalize=False, 
                 random_splits=False, random_testing_classes=34, only_noums_and_verbs=False, file_objects=None, 
                 objects_threshold=0.0, elaborative_descriptions=None):
        
        self.name = name
        self.action_classes = []
        self.samples = []
        self.training_set = []
        self.testing_set = []

        print("Loading data...")
        information_data = load_class_sentences(dataset_info, preprocess_embedder, min_class_sentence_len, max_class_sentences)        
        
        if elaborative_descriptions is not None:
            elab_descriptions = json.loads(open(elaborative_descriptions, "r").read()) ## from elaborative rehearshel
        else:
            elab_descriptions = []


        print("Embedding the semantic space...")
        with open(file_classes) as f:
            lines = f.readlines()
            for id, line in enumerate(lines):                
                line = line[:-1]
                line = preprocess_sentence(line)
                class_name = line
                label = process_label(line)
                class_sentences = []                
                for info in information_data[:]:
                    x = info                                       
                    if label == process_label(x[0]):
                        sent = preprocess_sentence(x[1])
                        if only_noums_and_verbs:
                            sent = get_noums_and_verbs(sent, return_sentence=True)                        
                        
                        class_sentences.append(sent + " "+ class_name)
            
                for desc in elab_descriptions:
                    if desc["word"].replace(" ","").lower() == label.lower():
                        elab_sents = proccess_paragraphs(desc["defn"], min_class_sentence_len)
                        for e in elab_sents:
                            class_sentences.append(e + class_name)
                        break
                    
                acc = ActionClass(id, class_name, label, class_sentences)              
                acc.compute_representations(zsar_embedder, normalize)
                self.action_classes.append(acc)        
        
        sentences_samples = []
        if load_samples:
            print("Loading samples...")
            predictions = load_samples_from_files(file_samples)

            if file_objects is not None:
                raise Exception("Object support is not implemented yet")
                #data_objects = json.loads(open(file_objects, "r").read())
            else:
                data_objects = {}

            for id, sample in enumerate(tqdm(list(predictions.keys()))):
                
                file_name = sample
                sentences = predictions[sample]["sentences"]                 

                # concatenating sentences --- in future, other procedures must be adopted
                sentence = " ".join(sentences).lower()                
                
                if only_noums_and_verbs:
                    sentence = get_noums_and_verbs(sentence, return_sentence=True)
                
                real_class = None
                for acc in self.action_classes: 
                    if acc.label in process_label(sample):
                        real_class = acc
                        break            
                if real_class is not None:
                    sentences_samples.append(sentence)
                    s = Sample(id, acc, sentence, file_name)
                    self.samples.append(s)
                else:
                    pass       
        
        sentence_embeddings = zsar_embedder.emb_sentence(sentences_samples)
        for i, emb in enumerate(sentence_embeddings):
            self.samples[i].embedding = emb.reshape(1,-1)

        
        splits = json.loads(open(file_zsl_splits).read())
        training_names = splits["training"]        
        testing_names = splits["testing"]
        

        if random_splits:
            names = training_names + testing_names
            random.shuffle(names)            
            training_names = names[0:-random_testing_classes]
            testing_names = names[-random_testing_classes:]      

        for t in training_names:
            for acc in self.action_classes:
                if acc.label == process_label(t):
                    self.training_set.append(acc)
        
        for t in testing_names:
            for acc in self.action_classes:
                if acc.label == process_label(t):
                    self.testing_set.append(acc)
        
        print("Dataset loaded!")

    
    def get_samples_by_class_label(self, class_label):
        samples = []
        for s in self.samples:
            if s.real_class.label == class_label:
                samples.append(s)                
        return samples
    
    def get_samples_by_class_id(self, class_id):
        samples = []
        for s in self.samples:
            if s.real_class.id == class_id:
                samples.append(s)                
        return samples
    

    def get_samples_by_split(self, split):        
        filtered_samples = []

        for sample in self.samples:
            for acc in split:
                if sample.real_class.id == acc.id:
                    filtered_samples.append(sample)
                    break
        return filtered_samples                

    def get_prototype_data(self, split, return_distances=False, mean_value=False, document_representation=False):
        X_prot = []
        y_prot = []
        sent_prot = []
        d = []                

        if mean_value == False:
            for acc in split:
                if document_representation:
                    X_prot.append(acc.document_representation)
                    y_prot.append(acc.id)
                else:
                    for i, sent_rep in enumerate(acc.sentences_representation):
                        X_prot.append(sent_rep)
                        y_prot.append(acc.id)
                        sent_prot.append(acc.sentences[i])
                        d.append(acc.sentence_distances[i])
        else:
            for acc in split: 
                X_prot.append(np.mean(np.concatenate(acc.sentences_representation, axis=0),axis=0).reshape(1,-1))
                y_prot.append(acc.id)                
                d.append(0)

        X_prot = np.concatenate(X_prot, axis=0)
        y_prot = np.asarray(y_prot, dtype=np.int32)
        if return_distances:
            return X_prot, y_prot, d, sent_prot
        else:
            return X_prot, y_prot, sent_prot


    def get_test_data(self, samples):

        X = []
        y = []
        for sample in samples:
            X.append(sample.embedding)
            y.append(sample.real_class.id)
        X = np.concatenate(X, axis=0)
        y = np.asarray(y, dtype=np.int32)
        return X, y
    
    def get_class_by_id(self, id):
        for c in self.action_classes:
            if c.id == id:
                return c
        return None

    def get_sample_by_id(self, id):
        for s in self.samples:
            if s.id == id:
                return s
        return None