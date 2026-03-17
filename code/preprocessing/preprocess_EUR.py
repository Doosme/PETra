input_folder = "../corpus_input/europarl_corpus/extracted/direct_trans/"
write_folder = "../corpus/"

import pandas
from nltk.tokenize import word_tokenize


def preprocess_eur_rawfile(read_lang_name, write_lang_str):
    data = pandas.read_csv(input_folder + "parallels_dir_ns_" + read_lang_name + ".tsv", sep='\t')
    original = data[data.columns[-2]]  # source language text
    target = data[data.columns[-1]]  # target language text

    with open(write_folder + "eur_" + write_lang_str + "_dir.txt", "w") as g:
        sent_id = 0
        g.write("sent_id\tsource\ttarget\tPOS (source)\tPOS (target)\talign\texpl_nulls (target)\tsimpl_nulls (source)\tentities")
        # split original and target into seperate lists
        for i in range(len(original)):
            src_sent = original[i]
            trg_sent = target[i]
            if type(src_sent) == str and type(trg_sent) == str and src_sent.strip() != "" and trg_sent.strip() != "":
                tokked_src = " ".join(word_tokenize(src_sent))
                tokked_trg = " ".join(word_tokenize(trg_sent))
                g.write("\n"+"eur_"+write_lang_str+"_dir_"+str(sent_id)+"\t"+tokked_src+"\t"+tokked_trg+"\t"+"\t"+"\t"+"\t"+"\t"+"\t")
                sent_id += 1


if __name__ == "__main__":
    LANG_LIST = [("EtoD", "en2de"), ("EtoF", "en2fr"), ("EtoI", "en2it"), ("EtoS", "en2es"), ("EtoP", "en2pt"),
                 ("DtoE", "de2en"), ("FtoE", "fr2en"), ("ItoE", "it2en"), ("StoE", "es2en"), ("PtoE", "pt2en")]

    for read_lang_name, write_lang_str in LANG_LIST:
        print("processing " + write_lang_str)
        preprocess_eur_rawfile(read_lang_name, write_lang_str)