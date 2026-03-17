input_folder = "../corpus_input/ted_multi"
write_folder = "../corpus/"

import datasets

def preprocess_ted_rawfile(src_lang, trg_lang):
    # load TED dataset
    lang_str = src_lang +"2"+ trg_lang
    dataset = datasets.load_dataset(input_folder)  # folder = TED_INPUT_PATH

    with open(write_folder + "ted_" + lang_str + ".txt", "w") as g:
        sent_id = 0
        g.write("sent_id\tsource\ttarget\tPOS (source)\tPOS (target)\talign\texpl_nulls (target)\tsimpl_nulls (source)\tentities")
        # filter data
        train_data = dataset["train"]
        test_data = dataset["test"]
        eval_data = dataset["validation"]
        data = datasets.concatenate_datasets([train_data, test_data, eval_data])

        # creates source and target lists
        for i in range(len(data)):
            lang = data[i]["translations"]["language"]
            sent = data[i]["translations"]["translation"]
            if src_lang in lang and trg_lang in lang:
                src_sent = sent[lang.index(src_lang)]
                trg_sent = sent[lang.index(trg_lang)]
                if src_sent.strip() == "" or trg_sent.strip() == "":
                    continue
                g.write("\n"+"ted_"+lang_str+"_"+str(sent_id)+"\t"+src_sent+"\t"+trg_sent+"\t"+"\t"+"\t"+"\t"+"\t"+"\t")
                sent_id += 1

if __name__ == "__main__":
    LANG_LIST = [("en","ar"),("en","de"),("en","fr"),("en","he"),("en","it"),("en","es"),("en","pt"),("en","el"),("en","fa"),("en","tr")]

    for src,trg in LANG_LIST:
        print("processing "+src+"2"+trg)
        preprocess_ted_rawfile(src, trg)

    #count lines:
    # find . -name 'ted*.txt' | xargs wc -l