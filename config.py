import os
import torch
from src.embedding import GloveEmbedder, Sent2vecEmbedder, TransformerEmbedder


class Config(object):

    def __init__(self, args):
        
        self.dataset_name = args.dataset_name
        if os.path.isfile(args.dataset_class_list):
            self.dataset_class_list = args.dataset_class_list
        else:
            raise Exception("Dataset class list file do not exists")
        
        if os.path.isfile(args.dataset_train_test_class_list):
            self.dataset_train_test_class_list = args.dataset_train_test_class_list
        else:
            raise Exception("Dataset train and text class lists file does not exist")

        if os.path.isdir(args.dataset_descriptions_dir):
            self.dataset_descriptions_dir = args.dataset_descriptions_dir
        else:
            raise Exception("Dataset descriptions dir dor not exists")

        self.embedder_for_semantic_preprocessing = args.embedder_for_semantic_preprocessing        
        self.min_words_per_sentence_description = args.min_words_per_sentence_description
        self.max_sentences_per_class = args.max_sentences_per_class
        self.concatenate_class_sentences = args.concatenate_class_sentences
        self.zsar_embedder_name = args.zsar_embedder_name
        self.normalize_embeddings = args.normalize_embeddings
        self.random_splits = args.random_splits
        self.random_testing_classes = args.random_testing_classes
        self.random_runs = args.random_runs
        self.use_elab_descriptions = args.use_elab_descriptions
        self.elab_descriptions_file = args.elab_descriptions_file
        self.k_neighbors = args.k_neighbors
        self.metric = args.metric
        self.observer_paths = args.observer_paths
        self.top_k = args.top_k
        self.gzsl = args.gzsl

        if self.embedder_for_semantic_preprocessing == "sent2vec":
            self.preprocess_embedder = Sent2vecEmbedder()
        elif self.embedder_for_semantic_preprocessing == "glove":
            self.preprocess_embedder = GloveEmbedder()
        else:
            self.preprocess_embedder = TransformerEmbedder(self.embedder_for_semantic_preprocessing)

        if self.zsar_embedder_name == "sent2vec":
            self.zsar_embedder = Sent2vecEmbedder()
        elif self.zsar_embedder_name == "glove":
            self.zsar_embedder = GloveEmbedder()
        else:
            self.zsar_embedder = TransformerEmbedder(self.zsar_embedder_name)

        if self.use_elab_descriptions:
            self.elaborative_descriptions = self.elab_descriptions_file
        else:
            self.elaborative_descriptions = None
        
        if self.random_splits:
            self.runs = self.random_runs
        else:
            self.runs = 1