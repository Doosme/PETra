import sys

from small_text import TransformersDataset, TransformerModelArguments, LeastConfidence, TransformerBasedClassificationFactory as TransformerFactory, PoolBasedActiveLearner

from transformers import AutoTokenizer
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score


DIR = "../active_learner_datasets/" #Path to Datasets Directory
POOL_DIR = "../" #todo: Path to Pool Directory




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
    all_labels = SEED_labels_labeled + [-1] * len(text_unlabeled)

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

    USED_indices = list(np.arange(len(SEED_text_labeled)))


    num_classes = 2
    model_args = TransformerModelArguments(transformer_model)
    clf_factory = TransformerFactory(model_args, num_classes, kwargs={'device': 'cuda', 'num_epochs': 10})


    query_strategy = LeastConfidence()


    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)
    initial_indices = np.arange(len(SEED_text_labeled))
    initial_labels = np.array(SEED_labels_labeled)
    active_learner.initialize_data(initial_indices, initial_labels)
    print("done initializing")

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
    print("Iteration Loop 0")
    print(f"Evaluation Accuracy: {acc:.4f}")
    print(f"Evaluation Precision: {prec:.4f}")
    print(f"Evaluation Recall: {rec:.4f}")
    print(f"Evaluation F1: {f1:.4f}")
    print(f"Evaluation F2: {f2:.4f}")
    print()

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
                        USED_indices.append(ted_ids_dict[ted_id])
                    elif tag == "FALSE":
                        cur_labels_labeled.append(0)
                        cur_list_ids.append(ted_ids_dict[ted_id])
                        USED_indices.append(ted_ids_dict[ted_id])
                    else:
                        print(tag)

        active_learner.indices_queried = np.array(cur_list_ids)
        active_learner.update(np.array(cur_labels_labeled))

        #scores
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
        print("Iteration Loop " + str(loop_idx))
        print(f"Evaluation Accuracy: {acc:.4f}")
        print(f"Evaluation Precision: {prec:.4f}")
        print(f"Evaluation Recall: {rec:.4f}")
        print(f"Evaluation F1: {f1:.4f}")
        print(f"Evaluation F2: {f2:.4f}")
        print()



    all_indices = np.arange(len(all_texts))

    pool_indices = np.array([i for i in all_indices if i not in USED_indices])

    pool_texts = [all_texts[i] for i in pool_indices]
    pool_ids = [ted_ids_list[i] for i in pool_indices]

    pool_dataset = TransformersDataset.from_arrays(
        pool_texts,
        y=[-1] * len(pool_texts),
        tokenizer=tokenizer,
        target_labels=np.array([0, 1]),
        max_length=512
    )

    # prediction
    predicted_labels = active_learner.classifier.predict(pool_dataset)
    predicted_probas = active_learner.classifier.predict_proba(pool_dataset)


    print("\nAutomatische Label-Vorschläge für den gesamten Pool:")
    print("ted_id\tsrc\ttrg\tlabel\tconfidence\tproba")
    for idx, text, label, proba, ted_id in zip(pool_indices, pool_texts, predicted_labels, predicted_probas, pool_ids):
        confidence = max(proba)
        print(str(ted_id) + "\t" + str(text.replace("[SEP]","\t")) + "\t" + str(label) + "\t" + str(confidence) + "\t" +str(proba))


def main_new_DE_ES():
    if "en2de" in CORPUS_NAME:
        base_CORPUS = "TED_en2de"
    elif "en2es"in CORPUS_NAME:
        base_CORPUS = "TED_en2es"
    else:
        raise ("Corpus Name not known: " + str(CORPUS_NAME))

    EVAL_text_labeled = []
    EVAL_labels_labeled = [] #1 for cultural, 0 for none
    SEED_text_labeled = []
    SEED_labels_labeled = [] #1 for cultural, 0 for none
    TRAIN_text_labeled = []
    TRAIN_labels_labeled = [] #1 for cultural, 0 for none

    text_unlabeled = []

    ted_ids_list = [] #list of ted_ids in the order of the seed + pool data
    ted_ids_dict = dict() #ted_ids: index in list


    USED_IDS = []
    #add initially discarded sentences
    with open(DIR + base_CORPUS + "_discarded_ids.tsv") as f:
        USED_IDS.extend(f.read().split("\n"))
    #add additional DISCARDs in further loops
    for loop_idx in range(1, CUR_LOOP_ID):
        with open(DIR + base_CORPUS + "_discarded_ids_LOOP"+str(loop_idx)+".tsv") as f:
            USED_IDS.extend(f.read().split("\n"))

    #transformer_model = 'bert-base-uncased'
    transformer_model = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)


    #read in SEED dataset
    first_line = True
    with open(DIR + base_CORPUS + "_SEED.tsv") as f:
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

    for loop_idx in range(1,CUR_LOOP_ID):
        first_line = True
        with open(DIR + base_CORPUS + "_TRAIN"+str(loop_idx)+".tsv") as f:
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
                        TRAIN_labels_labeled.append(-1)
                        TRAIN_text_labeled.append(src + tokenizer.sep_token + trg)
                        ted_ids_dict[ted_id] = len(ted_ids_list)
                        ted_ids_list.append(ted_id)
                    elif tag == "FALSE":
                        TRAIN_labels_labeled.append(-1)
                        TRAIN_text_labeled.append(src + tokenizer.sep_token + trg)
                        ted_ids_dict[ted_id] = len(ted_ids_list)
                        ted_ids_list.append(ted_id)
                    else:
                        print(tag)


    #read in EVAL dataset
    first_line = True
    with open(DIR + base_CORPUS + "_EVAL.tsv") as f:
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
    all_labels = SEED_labels_labeled + [-1] * len(text_unlabeled)

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

    USED_indices = list(np.arange(len(SEED_text_labeled)))


    #active learning configuration
    num_classes = 2
    model_args = TransformerModelArguments(transformer_model)
    clf_factory = TransformerFactory(model_args, num_classes, kwargs={'device': 'cuda', 'num_epochs': 10})


    query_strategy = LeastConfidence()


    #initialization
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)
    initial_indices = np.arange(len(SEED_text_labeled))
    initial_labels = np.array(SEED_labels_labeled)
    active_learner.initialize_data(initial_indices, initial_labels)
    print("done initializing")

    #initial scores
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
    print("Iteration Loop 0")
    print(f"Evaluation Accuracy: {acc:.4f}")
    print(f"Evaluation Precision: {prec:.4f}")
    print(f"Evaluation Recall: {rec:.4f}")
    print(f"Evaluation F1: {f1:.4f}")
    print(f"Evaluation F2: {f2:.4f}")
    print()

    #update with training sets
    for loop_idx in range(1,CUR_LOOP_ID):
        cur_labels_labeled = []
        cur_list_ids = []

        with open(DIR + base_CORPUS + "_TRAIN"+str(loop_idx)+".tsv") as f:
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
                        USED_indices.append(ted_ids_dict[ted_id])
                    elif tag == "FALSE":
                        cur_labels_labeled.append(0)
                        cur_list_ids.append(ted_ids_dict[ted_id])
                        USED_indices.append(ted_ids_dict[ted_id])
                    else:
                        print(tag)

        active_learner.indices_queried = np.array(cur_list_ids)
        active_learner.update(np.array(cur_labels_labeled))

        #scores
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
        print("Iteration Loop " + str(loop_idx))
        print(f"Evaluation Accuracy: {acc:.4f}")
        print(f"Evaluation Precision: {prec:.4f}")
        print(f"Evaluation Recall: {rec:.4f}")
        print(f"Evaluation F1: {f1:.4f}")
        print(f"Evaluation F2: {f2:.4f}")
        print()


    #active learning loop
    all_indices = np.arange(len(all_texts))

    pool_indices = np.array([i for i in all_indices if i not in USED_indices])

    pool_texts = [all_texts[i] for i in pool_indices]
    pool_ids = [ted_ids_list[i] for i in pool_indices]

    pool_dataset = TransformersDataset.from_arrays(
        pool_texts,
        y=[-1] * len(pool_texts),
        tokenizer=tokenizer,
        target_labels=np.array([0, 1]),
        max_length=512
    )

    # prediction
    predicted_labels = active_learner.classifier.predict(pool_dataset)
    predicted_probas = active_learner.classifier.predict_proba(pool_dataset)


    print("\nAutomatische Label-Vorschläge für den gesamten Pool:")
    print("ted_id\tsrc\ttrg\tlabel\tconfidence\tproba")
    for idx, text, label, proba, ted_id in zip(pool_indices, pool_texts, predicted_labels, predicted_probas, pool_ids):
        confidence = max(proba)
        print(str(ted_id) + "\t" + str(text.replace("[SEP]","\t")) + "\t" + str(label) + "\t" + str(confidence) + "\t" +str(proba))



if __name__ == "__main__":
    print("#### ANNOTATE FULL CORPUS ####")

    CORPUS_NAME = sys.argv[1]
    CUR_LOOP_ID = int(sys.argv[2])

    print(CORPUS_NAME + " (" + str(CUR_LOOP_ID) + ")")

    main_new_DE_ES() #includes TRAIN dataset
    main() #does not includee TRAIN dataset


