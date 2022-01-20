import argparse
import time
import numpy as np
from model.dataset import Dataset
from pprint import pprint
from config import Config
from sklearn.neighbors import NearestNeighbors


def evaluate_top_k(X, top_k, knn, y_prot, y_test):
    _, indexes = knn.kneighbors(X, n_neighbors=top_k)
    cont_top = 0
    for i, neighs in enumerate(indexes):
        for n in neighs:
            if y_prot[n] == y_test[i]:
                cont_top = cont_top + 1
                break
    return cont_top


def harmonic_mean(x, y):
    return (2 * x * y) / (x + y)


def print_summary(accs, title=""):
    print("="*60)
    print("Summary - "+title)
    for i, k in enumerate(cfg.top_k):
        x = np.asarray([acc[i] for acc in accs])
        print(
            f"top ({k})\tmean: {round(np.mean(x),2)} std: {round(np.std(x),2)}")
    print("="*60)


def zsar(cfg):

    accs = []
    seen_accs = []
    unseen_accs = []
    for run in range(cfg.runs):
        print(f"Run {run+1} of {cfg.runs}.\n\n")

        dataset = Dataset(cfg.dataset_name,
                          cfg.dataset_class_list,
                          cfg.dataset_train_test_class_list,
                          cfg.dataset_descriptions_dir,
                          cfg.min_words_per_sentence_description,
                          cfg.max_sentences_per_class,
                          cfg.observer_paths,
                          cfg.preprocess_embedder,
                          cfg.zsar_embedder,
                          random_splits=cfg.random_splits,
                          random_testing_classes=cfg.random_testing_classes,
                          normalize=cfg.normalize_embeddings,
                          elaborative_descriptions=cfg.elaborative_descriptions)

        print("Classifying...")
        if not cfg.gzsl:

            samples = dataset.get_samples_by_split(dataset.testing_set)
            print(f"Testing set has {len(dataset.testing_set)} classes")
            print(f"Testing set has {len(samples)} samples")
            X_prot, y_prot, _ = dataset.get_prototype_data(
                dataset.testing_set, document_representation=cfg.concatenate_class_sentences)
            X_test, y_test = dataset.get_test_data(samples)
            knn = NearestNeighbors(
                n_neighbors=cfg.k_neighbors, radius=1.0, metric=cfg.metric)  # 0.4
            knn.fit(X_prot, y_prot)

            t_accs = []
            for t_k in cfg.top_k:
                cont_top = evaluate_top_k(X_test, t_k, knn, y_prot, y_test)
                t_accs.append(100 * cont_top/y_test.shape[0])
                print(
                    f"Number of correctly classified (top {t_k}): {cont_top}")
                print(
                    f"Accuracy (top {t_k}): {round(100 * cont_top/y_test.shape[0],2)}\n")
            accs.append(t_accs)

        else:
            samples_training = dataset.get_samples_by_split(
                dataset.training_set)
            samples_testing = dataset.get_samples_by_split(dataset.testing_set)
            X_prot, y_prot, _ = dataset.get_prototype_data(
                dataset.training_set + dataset.testing_set, document_representation=cfg.concatenate_class_sentences)
            X_test_tr, y_test_tr = dataset.get_test_data(samples_training)
            X_test_te, y_test_te = dataset.get_test_data(samples_testing)

            knn = NearestNeighbors(
                n_neighbors=cfg.k_neighbors, radius=1.0, metric=cfg.metric)
            knn.fit(X_prot, y_prot)

            t_accs_tr = []
            for t_k in cfg.top_k:
                cont_top = evaluate_top_k(
                    X_test_tr, t_k, knn, y_prot, y_test_tr)
                t_accs_tr.append(100 * cont_top/y_test_tr.shape[0])
                print(
                    f"Number of correctly classified - seen (top {t_k}): {cont_top}")
                print(
                    f"Accuracy - seen (top {t_k}): {round(100 * cont_top/y_test_tr.shape[0],2)}\n")

            t_accs_te = []
            for t_k in cfg.top_k:
                cont_top = evaluate_top_k(
                    X_test_te, t_k, knn, y_prot, y_test_te)
                t_accs_te.append(100 * cont_top/y_test_te.shape[0])
                print(
                    f"Number of correctly classified - unseen (top {t_k}): {cont_top}")
                print(
                    f"Accuracy - unseen (top {t_k}): {round(100 * cont_top/y_test_te.shape[0],2)}\n")

            g_accs = []
            for i in range(len(t_accs_tr)):
                g_accs.append(harmonic_mean(t_accs_tr[i], t_accs_te[i]))

            seen_accs.append(t_accs_tr)
            unseen_accs.append(t_accs_te)
            accs.append(g_accs)

    if not cfg.gzsl:
        print_summary(accs, "ZSL")
    else:
        print_summary(seen_accs, "GZSL (seen)")
        print_summary(unseen_accs, "GZSL (unseen)")
        print_summary(accs, "GZSL (harmonic mean)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ZSAR Through Scene Descriptions")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_class_list", type=str,
                        help="File containing the class names list for the dataset")
    parser.add_argument("--dataset_train_test_class_list", type=str,
                        help="File contaning the training and testing class lists. It is used if you do not choose random splits option")
    parser.add_argument("--dataset_descriptions_dir", type=str,
                        help="Directory containing the files with descriptions for each class")
    parser.add_argument("--embedder_for_semantic_preprocessing", type=str, default="paraphrase-MiniLM-L6-v2",
                        choices=["glove", "sent2vec", "paraphrase-MiniLM-L6-v2", "paraphrase-distilroberta-base-v2"])
    parser.add_argument("--min_words_per_sentence_description", type=int,
                        default=15, help="Minimum length for semantic descriptive sentences")
    parser.add_argument("--max_sentences_per_class", type=int, default=10,
                        help="Maximum number of descriptive sentences for each class")
    parser.add_argument("--concatenate_class_sentences",
                        dest="concatenate_class_sentences", action="store_true", default=False)
    parser.add_argument("--zsar_embedder_name", type=str, default="paraphrase-distilroberta-base-v2",
                        choices=["glove", "sent2vec", "paraphrase-MiniLM-L6-v2", "paraphrase-distilroberta-base-v2"])
    parser.add_argument("--dont_normalize_embeddings",
                        dest="normalize_embeddings", action="store_false", default=True)
    parser.add_argument("--random_splits", dest="random_splits",
                        action="store_true", default=False)
    parser.add_argument("--random_testing_classes", type=int,
                        default=34, help="Number of the classes used in testing.")
    parser.add_argument("--random_runs", type=int, default=10,
                        help="Number of the random runs with random splits.")
    parser.add_argument("--use_elab_descriptions",
                        dest="use_elab_descriptions", action="store_true", default=False)
    parser.add_argument("--elab_descriptions_file", type=str,
                        help="File containing the elaborative descriptions")

    # k-nn configuration
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--top_k", type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["euclidean", "cosine"])
    # observers
    parser.add_argument('--observer_paths', type=str,
                        nargs='+', default=['', ''])

    # mode
    parser.add_argument("--gzsl", dest="gzsl",
                        action="store_true", default=False)

    args = parser.parse_args()
    pprint(vars(args))
    cfg = Config(args)

    start = time.time()
    zsar(cfg)
    print(f"Time taken: {time.time()-start} sec")
    print(f"Finish!")