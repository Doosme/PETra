import sys

from small_text import TransformersDataset, TransformerModelArguments, LeastConfidence, TransformerBasedClassificationFactory as TransformerFactory, PoolBasedActiveLearner
from small_text.query_strategies import QueryStrategy

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

import inspect
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import MiniBatchKMeans

DIR = "../active_learner_datasets/" #Path to Datasets Directory
POOL_DIR = "../" #todo: Path to Pool Directory

def _extract_n_from_args_kwargs(args, kwargs, default=10):
    if 'num_samples' in kwargs:
        return int(kwargs.get('num_samples'))
    if 'n' in kwargs:
        return int(kwargs.get('n'))
    if len(args) >= 1:
        return int(args[0])
    return int(default)

def _get_indices_and_labels(clf, dataset, kwargs):
    if 'indices_labeled' in kwargs and kwargs['indices_labeled'] is not None:
        indices_labeled = np.array(kwargs['indices_labeled'], dtype=int)
    else:
        if hasattr(clf, "indices_labeled") and getattr(clf, "indices_labeled") is not None:
            indices_labeled = np.array(getattr(clf, "indices_labeled"), dtype=int)
        elif hasattr(dataset, "labeled_indices") and getattr(dataset, "labeled_indices") is not None:
            indices_labeled = np.array(getattr(dataset, "labeled_indices"), dtype=int)
        else:
            indices_labeled = np.array([], dtype=int)

    if 'labels_labeled' in kwargs and kwargs['labels_labeled'] is not None:
        labels_labeled = np.array(kwargs['labels_labeled'], dtype=int)
    else:
        if hasattr(clf, "y") and getattr(clf, "y") is not None:
            labels_labeled = np.array(getattr(clf, "y"))
        elif hasattr(dataset, "y") and getattr(dataset, "y") is not None:
            labels_labeled = np.array(getattr(dataset, "y"))
        else:
            labels_labeled = np.array([], dtype=int)

    if len(indices_labeled) == 0 and len(labels_labeled) > 0:
        indices_labeled = np.where(labels_labeled != -1)[0]

    return indices_labeled, labels_labeled


def _parse_indices_and_n(dataset, args, kwargs, default_n=10):
    if 'indices_unlabeled' in kwargs:
        indices_unlabeled = np.array(kwargs['indices_unlabeled'], dtype=int)
    elif len(args) >= 1 and isinstance(args[0], (list, tuple, np.ndarray)):
        indices_unlabeled = np.array(args[0], dtype=int)
    else:
        try:
            indices_unlabeled = np.arange(len(dataset))
        except Exception:
            indices_unlabeled = np.array([], dtype=int)

    if 'num_samples' in kwargs:
        n = int(kwargs['num_samples'])
    elif 'n' in kwargs:
        n = int(kwargs['n'])
    else:
        n = None
        for a in args:
            if isinstance(a, (int, np.integer)):
                n = int(a)
                break
        if n is None:
            n = int(default_n)

    return indices_unlabeled, n


def call_query(strategy, clf, dataset, indices_unlabeled, n, **kwargs):
    indices_unlabeled = np.array(indices_unlabeled, dtype=int)
    n = int(n)

    indices_labeled, labels_labeled = _get_indices_and_labels(clf, dataset, kwargs)

    sig = inspect.signature(strategy.query)
    params = list(sig.parameters.keys())
    pos_args = [clf, dataset]
    kw = dict(kwargs)

    if 'indices_unlabeled' in params:
        kw['indices_unlabeled'] = indices_unlabeled
    else:
        third_param = params[2] if len(params) > 2 else None
        if third_param not in ('n', 'num_samples'):
            pos_args.append(indices_unlabeled)

    if 'indices_labeled' in params:
        kw['indices_labeled'] = indices_labeled
    if 'labels_labeled' in params:
        kw['labels_labeled'] = labels_labeled
    if 'y' in params:
        kw['y'] = labels_labeled

    if 'n' in params:
        kw['n'] = n
    elif 'num_samples' in params:
        kw['num_samples'] = n
    else:
        kw['n'] = n

    try:
        result = strategy.query(*pos_args, **kw)
        return np.array(result, dtype=int)
    except TypeError:
        try:
            pos_args2 = [clf, dataset, np.array(indices_unlabeled, dtype=int),
                         np.array(indices_labeled, dtype=int), np.array(labels_labeled)]
            if 'n' in params:
                res = strategy.query(*pos_args2, n=n)
            elif 'num_samples' in params:
                res = strategy.query(*pos_args2, num_samples=n)
            else:
                res = strategy.query(*pos_args2, n=n)
            return np.array(res, dtype=int)
        except Exception:
            try:
                dataset_slice = dataset[indices_unlabeled]
                if 'n' in params:
                    res = strategy.query(clf, dataset_slice, n=n)
                elif 'num_samples' in params:
                    res = strategy.query(clf, dataset_slice, num_samples=n)
                else:
                    res = strategy.query(clf, dataset_slice, n=n)
                return np.array(res, dtype=int)
            except Exception:
                raise

class EmbeddingKMeansCustom(QueryStrategy):
    def __init__(self, embeddings, n_clusters=None, random_state=42):
        self.embeddings = np.array(embeddings)
        self.n_clusters = n_clusters
        self.random_state = random_state

    def query(self, clf, dataset, *args, **kwargs):
        indices_unlabeled, n = _parse_indices_and_n(dataset, args, kwargs, default_n=10)
        if len(indices_unlabeled) == 0:
            return np.array([], dtype=int)

        emb_unlabeled = self.embeddings[indices_unlabeled]

        n_clusters = self.n_clusters or max(1, min(len(indices_unlabeled), n))

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state)
        km.fit(emb_unlabeled)
        cluster_centers = km.cluster_centers_
        cluster_labels = km.labels_

        selected = []
        for c in range(n_clusters):
            cluster_indices = np.where(cluster_labels == c)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_emb = emb_unlabeled[cluster_indices]
            dists = np.linalg.norm(cluster_emb - cluster_centers[c], axis=1)
            best_idx_local = cluster_indices[np.argmin(dists)]
            selected.append(indices_unlabeled[best_idx_local])

        if len(selected) < n:
            remaining = list(set(indices_unlabeled) - set(selected))
            np.random.shuffle(remaining)
            selected.extend(remaining[:n - len(selected)])

        return np.array(selected[:n], dtype=int)


class DiverseSeedPositiveVariants(QueryStrategy):
    def __init__(self, all_texts=None, seed_pos_indices=None, embeddings=None, top_k_variants=5, fraction=1.0):
        self.all_texts = all_texts
        self.seed_pos_indices = np.array(seed_pos_indices, dtype=int) if seed_pos_indices is not None else None
        self.embeddings = embeddings
        self.top_k_variants = top_k_variants
        self.fraction = fraction

    def query(self, clf, dataset, *args, **kwargs):
        indices_unlabeled, n = _parse_indices_and_n(dataset, args, kwargs, default_n=10)

        if self.seed_pos_indices is None or len(self.seed_pos_indices) == 0:
            print("️ Keine Seed-Positiven angegeben, wähle zufällig.")
            choice = np.random.choice(indices_unlabeled, size=min(n, len(indices_unlabeled)), replace=False)
            return np.array(choice, dtype=int)

        try:
            emb_unl = clf.embed(dataset[indices_unlabeled])
            emb_seed = clf.embed(dataset[self.seed_pos_indices])
        except Exception:
            if self.embeddings is not None:
                emb_unl = self.embeddings[indices_unlabeled]
                emb_seed = self.embeddings[self.seed_pos_indices]
            else:
                raise RuntimeError("Keine Embeddings verfügbar für DiverseSeedPositiveVariants")

        pos_dists = cosine_distances(emb_seed)
        diversity_scores = pos_dists.sum(axis=1)
        num_diverse = max(1, int(len(self.seed_pos_indices) * self.fraction))
        diverse_indices_seed = np.array(self.seed_pos_indices)[np.argsort(-diversity_scores)[:num_diverse]]

        selected_indices = []
        for idx in diverse_indices_seed:
            seed_emb = emb_seed[np.where(self.seed_pos_indices == idx)[0][0]]
            dists_to_pool = cosine_distances([seed_emb], emb_unl)[0]
            nearest = np.argsort(dists_to_pool)[:self.top_k_variants]
            selected_indices.extend(indices_unlabeled[nearest])
        selected_indices = list(dict.fromkeys(selected_indices))

        if len(selected_indices) > n:
            selected_indices = np.random.choice(selected_indices, size=n, replace=False)

        return np.array(selected_indices, dtype=int)

class DiversePositiveVariants(QueryStrategy):
    def __init__(self, all_texts=None, embeddings=None, top_k_variants=5, fraction=1.0):
        self.all_texts = all_texts
        self.embeddings = embeddings
        self.top_k_variants = top_k_variants
        self.fraction = fraction

    def query(self, clf, dataset, *args, **kwargs):
        indices_unlabeled, n = _parse_indices_and_n(dataset, args, kwargs, default_n=10)

        print("n used in DiversePositiveVariants.query(): " + str(n))

        indices_labeled, labels_labeled = _get_indices_and_labels(clf, dataset, kwargs)
        pos_indices = [i for i, y in zip(indices_labeled, labels_labeled) if int(y) == 1]

        print(indices_labeled)
        print(labels_labeled)
        print(type(labels_labeled[0]))

        print()
        print(pos_indices)
        print([dataset[i] for i in pos_indices])
        print()

        if len(pos_indices) == 0:
            choice = np.random.choice(indices_unlabeled, size=min(n, len(indices_unlabeled)), replace=False)
            return np.array(choice, dtype=int)

        try:
            emb_unl = clf.embed(dataset[indices_unlabeled])
            emb_pos = clf.embed(dataset[pos_indices])
        except Exception:
            if self.embeddings is not None:
                emb_unl = self.embeddings[indices_unlabeled]
                emb_pos = self.embeddings[pos_indices]
            else:
                raise RuntimeError("Keine Embeddings verfügbar für DiversePositiveVariants")

        pos_dists = cosine_distances(emb_pos)
        diversity_scores = pos_dists.sum(axis=1)
        num_diverse = max(1, int(len(pos_indices) * self.fraction))
        diverse_indices_pos = np.array(pos_indices)[np.argsort(-diversity_scores)[:num_diverse]]

        selected_indices = []
        for i, idx in enumerate(diverse_indices_pos):
            pos_emb = emb_pos[np.where(np.array(pos_indices) == idx)[0][0]]
            dists_to_pool = cosine_distances([pos_emb], emb_unl)[0]
            nearest = np.argsort(dists_to_pool)[:self.top_k_variants]
            selected_indices.extend(indices_unlabeled[nearest])

        selected_indices = list(dict.fromkeys(selected_indices))

        if len(selected_indices) > n:
            selected_indices = np.random.choice(selected_indices, size=n, replace=False)

        return np.array(selected_indices, dtype=int)


class NearestSeedPositiveNeighbors(QueryStrategy):
    def __init__(self, all_texts, seed_pos_indices=None, embeddings=None, top_k=50, fraction=1.0):
        self.all_texts = all_texts
        self.seed_pos_indices = np.array(seed_pos_indices, dtype=int) if seed_pos_indices is not None else None
        self.embeddings = embeddings
        self.top_k = top_k
        self.fraction = fraction

    def query(self, clf, dataset, *args, **kwargs):
        indices_unlabeled, n = _parse_indices_and_n(dataset, args, kwargs, default_n=10)

        if self.seed_pos_indices is None or len(self.seed_pos_indices) == 0:
            if len(indices_unlabeled) == 0:
                return np.array([], dtype=int)
            choice = np.random.choice(indices_unlabeled, size=min(n, len(indices_unlabeled)), replace=False)
            return np.array(choice, dtype=int)

        try:
            emb_unl = clf.embed(dataset[indices_unlabeled])
            emb_seed = clf.embed(dataset[self.seed_pos_indices])
        except Exception:
            if self.embeddings is not None:
                emb_unl = self.embeddings[indices_unlabeled]
                emb_seed = self.embeddings[self.seed_pos_indices]
            else:
                raise RuntimeError("Keine Embeddings verfügbar für NearestSeedPositiveNeighbors")

        dists = cosine_distances(emb_unl, emb_seed)
        nearest_dist = np.min(dists, axis=1)
        order = np.argsort(nearest_dist)
        selected_local = order[:min(n, len(order))]

        return np.array(indices_unlabeled)[selected_local]


class NearestPositiveNeighbors(QueryStrategy):
    def __init__(self, all_texts, embeddings=None, top_k=50, fraction=1.0):
        self.all_texts = all_texts
        self.embeddings = embeddings
        self.top_k = top_k
        self.fraction = fraction

    def query(self, clf, dataset, *args, **kwargs):
        indices_unlabeled, n = _parse_indices_and_n(dataset, args, kwargs, default_n=10)


        print("n used in NearestPositiveNeighbors.query(): " + str(n))

        indices_labeled, labels_labeled = _get_indices_and_labels(clf, dataset, kwargs)

        pos_indices = [i for i, y in zip(indices_labeled, labels_labeled) if int(y) == 1]
        if len(pos_indices) == 0:
            if len(indices_unlabeled) == 0:
                return np.array([], dtype=int)
            choice = np.random.choice(indices_unlabeled, size=min(n, len(indices_unlabeled)), replace=False)
            return np.array(choice, dtype=int)

        try:
            emb_unl = clf.embed(dataset[indices_unlabeled])
            emb_pos = clf.embed(dataset[pos_indices])
        except Exception:
            if self.embeddings is not None:
                emb_unl = self.embeddings[indices_unlabeled]
                emb_pos = self.embeddings[pos_indices]
            else:
                raise

        dists = cosine_distances(emb_unl, emb_pos)
        nearest_dist = np.min(dists, axis=1)
        order = np.argsort(nearest_dist)
        selected_local = order[:min(n, len(order))]
        return np.array(indices_unlabeled)[selected_local]


class HighConfidencePositives(QueryStrategy):
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def query(self, clf, dataset, *args, **kwargs):
        indices_unlabeled, n = _parse_indices_and_n(dataset, args, kwargs, default_n=10)
        if len(indices_unlabeled) == 0:
            return np.array([], dtype=int)

        print("n used in HighConfidencePositives.query(): " + str(n))

        probs = clf.predict_proba(dataset[indices_unlabeled])
        pos_probs = probs[:, 1]

        candidate_mask = pos_probs >= self.threshold
        candidate_indices = indices_unlabeled[candidate_mask]

        if len(candidate_indices) == 0:
            order = np.argsort(-pos_probs)
            selected = indices_unlabeled[order[:min(n, len(order))]]
            return np.array(selected, dtype=int)

        n_select = min(n, len(candidate_indices))
        selected = np.random.choice(candidate_indices, size=n_select, replace=False)

        return np.array(selected, dtype=int)



class CombinedQueryStrategy(QueryStrategy):
    def __init__(self, strategy_a, strategy_b, fraction_a=0.7):
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.fraction_a = fraction_a

    def query(self, clf, dataset, *args, **kwargs):
        indices_unlabeled, n = _parse_indices_and_n(dataset, args, kwargs, default_n=10)

        n_a = int(round(n * self.fraction_a))
        n_b = max(0, n - n_a)
        print("n: "+str(n))
        print("n_a: "+str(n_a))
        print("n_b: "+str(n_b))


        indices_labeled, labels_labeled = _get_indices_and_labels(clf, dataset, kwargs)

        print('sig strategy_a:', inspect.signature(self.strategy_a.query))
        print('sig strategy_b:', inspect.signature(self.strategy_b.query))

        # Strategy A
        sel_a = call_query(self.strategy_a, clf, dataset, indices_unlabeled, n_a,
                           indices_labeled=indices_labeled,
                           labels_labeled=labels_labeled)

        # remaining Pool
        remaining = np.array(list(set(indices_unlabeled) - set(sel_a)), dtype=int)
        if len(remaining) == 0 or n_b == 0:
            return sel_a[:n]

        # Strategy B
        sel_b = call_query(self.strategy_b, clf, dataset, remaining, n_b,
                           indices_labeled=indices_labeled,
                           labels_labeled=labels_labeled)

        result = np.concatenate([sel_a, sel_b]).astype(int)
        return result[:n]

class CombinedQueryStrategy3(QueryStrategy):
    def __init__(self, strategy_a, strategy_b, strategy_c, fraction_a=0.5, fraction_b=0.3):
        assert 0 <= fraction_a <= 1
        assert 0 <= fraction_b <= 1
        assert fraction_a + fraction_b <= 1
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.strategy_c = strategy_c
        self.fraction_a = fraction_a
        self.fraction_b = fraction_b

    def query(self, clf, dataset, *args, **kwargs):
        indices_unlabeled, n = _parse_indices_and_n(dataset, args, kwargs, default_n=10)
        if len(indices_unlabeled) == 0:
            return np.array([], dtype=int)

        indices_labeled, labels_labeled = _get_indices_and_labels(clf, dataset, kwargs)

        # amount per strategy
        n_a = int(round(n * self.fraction_a))
        n_b = int(round(n * self.fraction_b))
        n_c = n - n_a - n_b

        print("n: "+str(n))
        print("n_a: "+str(n_a))
        print("n_b: "+str(n_b))
        print("n_c: "+str(n_c))

        # Strategy A
        sel_a = call_query(self.strategy_a, clf, dataset, indices_unlabeled, n_a,
                           indices_labeled=indices_labeled,
                           labels_labeled=labels_labeled)

        # remaining Pool
        remaining = np.array(list(set(indices_unlabeled) - set(sel_a)), dtype=int)
        if len(remaining) == 0 or n_b + n_c == 0:
            return sel_a[:n]

        # Strategy B
        sel_b = call_query(self.strategy_b, clf, dataset, remaining, n_b,
                           indices_labeled=indices_labeled,
                           labels_labeled=labels_labeled)

        # remaining Pool for C
        remaining2 = np.array(list(set(remaining) - set(sel_b)), dtype=int)
        if len(remaining2) == 0 or n_c == 0:
            return np.concatenate([sel_a, sel_b])[:n]

        # Strategy C
        sel_c = call_query(self.strategy_c, clf, dataset, remaining2, n_c,
                           indices_labeled=indices_labeled,
                           labels_labeled=labels_labeled)

        result = np.concatenate([sel_a, sel_b, sel_c]).astype(int)
        return result[:n]



def main():
    EVAL_text_labeled = []
    EVAL_labels_labeled = [] #1 for cultural, 0 for none
    SEED_text_labeled = []
    SEED_labels_labeled = [] #1 for cultural, 0 for none

    text_unlabeled = []

    ted_ids_list = [] #list of ted_ids in the order of the seed + pool data
    ted_ids_dict = dict() #ted_ids: index in list

    USED_IDS = []
    #add initially discarded sentences
    with open(DIR + CORPUS_NAME + "_discarded_ids.tsv") as f:
        USED_IDS.extend(f.read().split("\n"))
    #add additional DISCARDs in further loops
    for loop_idx in range(1, CUR_LOOP_ID):
        with open(DIR + CORPUS_NAME + "_discarded_ids_LOOP"+str(loop_idx)+".tsv") as f:
            USED_IDS.extend(f.read().split("\n"))


    #transformer_model = 'bert-base-uncased'
    transformer_model = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)


    #read in SEED dataset
    first_line = True
    with open(DIR + CORPUS_NAME + "_SEED.tsv") as f:
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.strip("\n")
            line_lst = line.split("\t")
            ted_id = line_lst[0]
            src = line_lst[1]
            trg = line_lst[2]
            tag = line_lst[3]
            if ted_id != "":
                USED_IDS.append(ted_id)
                if tag == "TRUE":
                    SEED_labels_labeled.append(1)
                    SEED_text_labeled.append(src +tokenizer.sep_token+ trg)
                    ted_ids_dict[ted_id] = len(ted_ids_list)
                    ted_ids_list.append(ted_id)
                elif tag == "FALSE":
                    SEED_labels_labeled.append(0)
                    SEED_text_labeled.append(src +tokenizer.sep_token+ trg)
                    ted_ids_dict[ted_id] = len(ted_ids_list)
                    ted_ids_list.append(ted_id)
                else:
                    print(tag)

    print("Seed distribution:", np.bincount(SEED_labels_labeled))

    #read in EVAL dataset
    first_line = True
    with open(DIR + CORPUS_NAME + "_EVAL.tsv") as f:
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.strip("\n")
            line_lst = line.split("\t")
            ted_id = line_lst[0]
            src = line_lst[1]
            trg = line_lst[2]
            tag = line_lst[3]
            if ted_id != "":
                USED_IDS.append(ted_id)
                if tag == "TRUE":
                    EVAL_labels_labeled.append(1)
                    EVAL_text_labeled.append(src +tokenizer.sep_token+ trg)
                elif tag == "FALSE":
                    EVAL_labels_labeled.append(0)
                    EVAL_text_labeled.append(src +tokenizer.sep_token+ trg)
                else:
                    print(tag)


    with open(POOL_DIR + CORPUS_NAME + "_full.txt") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.strip("\n")
            line_lst = line.split("\t")
            ted_id = line_lst[0]
            src = line_lst[1]
            trg = line_lst[2]
            if ted_id != "" and ted_id not in USED_IDS:
                text_unlabeled.append(src + tokenizer.sep_token + trg)
                ted_ids_dict[ted_id] = len(ted_ids_list)
                ted_ids_list.append(ted_id)


    all_texts = SEED_text_labeled + text_unlabeled
    all_labels = SEED_labels_labeled + [-1] * len(text_unlabeled) #-1: placeholder for unlabeled

    src_all = []
    trg_all = []
    for text in SEED_text_labeled + text_unlabeled:
        if "[SEP]" in text:
            src, trg = text.split("[SEP]")
        else:
            src, trg = text, ""
        src_all.append(src.strip())
        trg_all.append(trg.strip())

    texts = list(zip(src_all, trg_all))

    dataset = TransformersDataset.from_arrays(
        texts,
        all_labels,
        tokenizer,
        target_labels=np.array([0, 1]),
        max_length=512
    )

    seed_pos_indices = [i for i, label in enumerate(SEED_labels_labeled) if label == 1]

    #active learning configuration
    num_classes = 2
    model_args = TransformerModelArguments(transformer_model)
    clf_factory = TransformerFactory(model_args, num_classes, kwargs={'device': 'cuda', 'num_epochs': 10})

    if CUR_LOOP_ID <= 4:
        if CUR_LOOP_ID % 3 == 1:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            sources, targets = [], []
            for text in all_texts:
                if "[SEP]" in text:
                    src, tgt = text.split("[SEP]")
                else:
                    src, tgt = text, ""
                sources.append(src.strip())
                targets.append(tgt.strip())
            emb_src = embedder.encode(sources, convert_to_numpy=True, show_progress_bar=True, batch_size=256,device='cuda')
            emb_tgt = embedder.encode(targets, convert_to_numpy=True, show_progress_bar=True, batch_size=256,device='cuda')
            embeddings = np.concatenate([emb_src, emb_tgt], axis=1)

            query_strategy = CombinedQueryStrategy3(
                strategy_a=HighConfidencePositives(threshold=0.65),
                strategy_b=EmbeddingKMeansCustom(embeddings=embeddings),
                strategy_c=LeastConfidence(),
                fraction_a=0.5,
                fraction_b=0.4
        )
        elif CUR_LOOP_ID % 3 == 2:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            sources, targets = [], []
            for text in all_texts:
                if "[SEP]" in text:
                    src, tgt = text.split("[SEP]")
                else:
                    src, tgt = text, ""
                sources.append(src.strip())
                targets.append(tgt.strip())
            emb_src = embedder.encode(sources, convert_to_numpy=True, show_progress_bar=True, batch_size=256, device='cuda')
            emb_tgt = embedder.encode(targets, convert_to_numpy=True, show_progress_bar=True, batch_size=256, device='cuda')
            embeddings = np.concatenate([emb_src, emb_tgt], axis=1)

            query_strategy = CombinedQueryStrategy3(
                strategy_a=HighConfidencePositives(threshold=0.8),
                strategy_b=NearestPositiveNeighbors(all_texts, embeddings, top_k=50, fraction=1.0),
                strategy_c=LeastConfidence(),
                fraction_a=0.3,
                fraction_b=0.2,
            )
        else:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            sources, targets = [], []
            for text in all_texts:
                if "[SEP]" in text:
                    src, tgt = text.split("[SEP]")
                else:
                    src, tgt = text, ""
                sources.append(src.strip())
                targets.append(tgt.strip())
            emb_src = embedder.encode(sources, convert_to_numpy=True, show_progress_bar=True, batch_size=256, device='cuda')
            emb_tgt = embedder.encode(targets, convert_to_numpy=True, show_progress_bar=True, batch_size=256, device='cuda')
            embeddings = np.concatenate([emb_src, emb_tgt], axis=1)

            query_strategy = CombinedQueryStrategy(
                strategy_a=DiverseSeedPositiveVariants(all_texts, seed_pos_indices, embeddings, top_k_variants=3,fraction=1.0),
                strategy_b=NearestSeedPositiveNeighbors(all_texts, seed_pos_indices, embeddings, top_k=50, fraction=1.0),
                fraction_a=0.7,
            )
    elif CUR_LOOP_ID <= 7:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        sources, targets = [], []
        for text in all_texts:
            if "[SEP]" in text:
                src, tgt = text.split("[SEP]")
            else:
                src, tgt = text, ""
            sources.append(src.strip())
            targets.append(tgt.strip())
        emb_src = embedder.encode(sources, convert_to_numpy=True, show_progress_bar=True, batch_size=256, device='cuda')
        emb_tgt = embedder.encode(targets, convert_to_numpy=True, show_progress_bar=True, batch_size=256, device='cuda')
        embeddings = np.concatenate([emb_src, emb_tgt], axis=1)

        query_strategy = CombinedQueryStrategy(
            strategy_a=DiversePositiveVariants(all_texts, embeddings, top_k_variants=3, fraction=1.0),
            strategy_b=LeastConfidence(),
            fraction_a=0.8,
        )

    elif CUR_LOOP_ID <= 8:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            sources, targets = [], []
            for text in all_texts:
                if "[SEP]" in text:
                    src, tgt = text.split("[SEP]")
                else:
                    src, tgt = text, ""
                sources.append(src.strip())
                targets.append(tgt.strip())
            emb_src = embedder.encode(sources, convert_to_numpy=True, show_progress_bar=True, batch_size=256,device='cuda')
            emb_tgt = embedder.encode(targets, convert_to_numpy=True, show_progress_bar=True, batch_size=256,device='cuda')
            embeddings = np.concatenate([emb_src, emb_tgt], axis=1)

            query_strategy = CombinedQueryStrategy3(
                strategy_a=HighConfidencePositives(threshold=0.65),
                strategy_b=NearestPositiveNeighbors(all_texts, embeddings, top_k=50, fraction=1.0),
                strategy_c=DiversePositiveVariants(all_texts, embeddings, top_k_variants=3, fraction=1.0),
                fraction_a=0.4,
                fraction_b=0.3
        )
    else:
        query_strategy = LeastConfidence()



    #initialization
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)
    initial_indices = np.arange(len(SEED_text_labeled))
    initial_labels = np.array(SEED_labels_labeled)
    active_learner.initialize_data(initial_indices, initial_labels)
    print("done initializing")

    #update with training sets
    for loop_idx in range(1,CUR_LOOP_ID):
        cur_labels_labeled = []
        cur_list_ids = []

        with open(DIR + CORPUS_NAME + "_TRAIN"+str(loop_idx)+".tsv") as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    continue
                line = line.strip("\n")
                line_lst = line.split("\t")
                ted_id = line_lst[0]
                tag = line_lst[3]
                if ted_id != "":
                    USED_IDS.append(ted_id)
                    if tag == "TRUE":
                        cur_labels_labeled.append(1)
                        cur_list_ids.append(ted_ids_dict[ted_id])
                    elif tag == "FALSE":
                        cur_labels_labeled.append(0)
                        cur_list_ids.append(ted_ids_dict[ted_id])
                    else:
                        print(tag)

        active_learner.indices_queried = np.array(cur_list_ids)
        active_learner.update(np.array(cur_labels_labeled))

    # predictions on eval
    eval_dataset = TransformersDataset.from_arrays(
        EVAL_text_labeled,
        EVAL_labels_labeled,
        tokenizer,
        target_labels=np.array([0, 1]),
        max_length=512
    )
    y_pred = active_learner.classifier.predict(eval_dataset)
    y_true = np.array(EVAL_labels_labeled)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    print()
    print("Iteration Loop " + str(CUR_LOOP_ID - 1) + " (" +str(NUM_OF_SAMPLES)+ ")")
    print(f"Evaluation Accuracy: {acc:.4f}")
    print(f"Evaluation Precision: {prec:.4f}")
    print(f"Evaluation Recall: {rec:.4f}")
    print(f"Evaluation F1: {f1:.4f}")
    print(f"Evaluation F2: {f2:.4f}")
    print()

    #active learning loop
    indices_queried = active_learner.query(num_samples=NUM_OF_SAMPLES)

    queried_texts = [all_texts[i] for i in list(indices_queried)]
    queried_ids = [ted_ids_list[i] for i in list(indices_queried)]

    queried_dataset = TransformersDataset.from_arrays(
        queried_texts,
        y=[-1] * len(queried_texts),
        tokenizer=tokenizer,
        target_labels=np.array([0, 1]),
        max_length=512
    )

    predicted_labels = active_learner.classifier.predict(queried_dataset)
    predicted_probas = active_learner.classifier.predict_proba(queried_dataset)

    print("\nAutomatische Label-Vorschläge:")
    print("ted_id\tsrc\ttrg\tlabel\tconfidence\tproba")
    for idx, text, label, proba, ted_id in zip(indices_queried, queried_texts, predicted_labels, predicted_probas, queried_ids):
        confidence = max(proba)
        print(str(ted_id) + "\t" + str(text.replace("[SEP]","\t")) + "\t" + str(label) + "\t" + str(confidence) + "\t" +str(proba))
        #print(f"\nTED-ID: {ted_id}")
        #print(f"Text: {text}")
        #print(f"Vorgeschlagenes Label: {label} (Confidence: {confidence:.2f})")
        #print(f"Label-Wahrscheinlichkeiten: {proba}")

    #print()
    #print("New Instances to Annotate (ids): ", indices_queried)
    #print("New Instances to Annotate (ted_ids): ", [ted_ids_list[index] for index in list(indices_queried)])


if __name__ == "__main__":
    CORPUS_NAME = sys.argv[1]
    NUM_OF_SAMPLES = int(sys.argv[2])
    CUR_LOOP_ID = int(sys.argv[3])
    main()
