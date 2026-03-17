import sys
import spacy


CORPUS_DIR = "../NEW_corpus/"

from util import choose_model

import simalign

DICT_FILE = "de.dict"

RELEVANT_ENT_TYPES = ["PERSON", "NORP", "FAC","ORG","GPE","LOC",
                      "PRODUCT", "EVENT", "WORK_OF_ART","LAW","LANGUAGE"] 

PUNCTUATION = [".", ",","-",";","?","!","&quot;"]

# adapted from https://github.com/jodaiber/semantic_compound_splitting
def load_dict(file):
    splits = {}
    with open(file) as f:
        for line in f:
            es = line.rstrip('\n').split(" ")
            w = es[0]
            w = w.lower()

            indices = map(lambda i: i.split(','), es[1:])

            splits[w] = []
            for from_, to, fug in indices:
                s, e = int(from_), int(to)
                # Don't use single character splits - just add to prev split
                if e - s == 1:
                    splits[w][-1][1] += 1
                else:
                    splits[w].append([s, e, fug])
    return splits


# adapted from https://github.com/jodaiber/semantic_compound_splitting
def split_word(w, splits):
    if w.lower() in splits:
        w_split = []
        for from_, to, fug in splits[w.lower()]:
            wordpart = w[from_:to-len(fug)]

            if w == w.title():
                wordpart = wordpart.title()
            elif w == w.upper():
                wordpart = wordpart.upper()

            w_split.append(wordpart)

        return " ".join(w_split)
    else:
        return w


def decompose_DE_trg(readfile, writefile, SRC_NLP, TRG_NLP):
    compound_splits = load_dict(DICT_FILE)
    aligner_SIM = simalign.SentenceAligner(matching_methods="f")  # f = forward; m = match

    with open(readfile) as f:
        with open(writefile, "w") as g:
            count = 0

            firstline = True
            data = f.read()
            data_list = data.split("\n")
            for line in data_list:
                if firstline:
                    g.write(line.strip())
                    firstline = False
                    continue

                line_l = line.split("\t")
                sent_id = line_l[0]
                src = line_l[1]
                trg = line_l[2]
                pos_src = line_l[3]
                pos_trg = line_l[4]
                ent_src = line_l[12]
                ent_trg = line_l[13]

                if count % 1000 == 0: print(count)

                #decompose trg
                trg_toks = trg.split(" ")
                trg_toks_decomp = [split_word(tok, splits=compound_splits) for tok in trg_toks]
                trg = " ".join(trg_toks_decomp)

                #realign trg with SIM
                aligned_SIM = aligner_SIM.get_word_aligns(src, trg)
                aligned_str = ""
                for src_align_id, trg_align_id in aligned_SIM["fwd"]:
                    aligned_str += str(src_align_id) + "-" + str(trg_align_id) + " "
                aligned_str_SIM = aligned_str.strip()

                #postag trg
                trg_doc = TRG_NLP(spacy.tokens.Doc(TRG_NLP.vocab, words=trg.split(" ")))
                trg_pos = [tok.pos_ for tok in trg_doc]  # (tok.text, tok.pos_)
                trg_pos_str = " ".join(trg_pos)

                # calculate nulls

                #write to file
                g.write("\n" + sent_id + "\t" + src + "\t" + trg + "\t" + pos_src + "\t" + trg_pos_str + "\t" + "\t" +aligned_str_SIM+ "\t" + "\t" + "\t" + "\t" + "\t" + "\t" +ent_src+ "\t"+ent_trg)

                count += 1


def decompose_DE_src(readfile, writefile, SRC_NLP, TRG_NLP):
    compound_splits = load_dict(DICT_FILE)
    aligner_SIM = simalign.SentenceAligner(matching_methods="f")  # f = forward; m = match

    with open(readfile) as f:
        with open(writefile, "w") as g:
            count = 0

            firstline = True
            data = f.read()
            data_list = data.split("\n")
            for line in data_list:
                if firstline:
                    g.write(line.strip())
                    firstline = False
                    continue

                line_l = line.split("\t")
                sent_id = line_l[0]
                src = line_l[1]
                trg = line_l[2]
                pos_src = line_l[3]
                pos_trg = line_l[4]
                ent_src = line_l[12]
                ent_trg = line_l[13]

                if count % 1000 == 0: print(count)

                #decompose src
                src_toks = src.split(" ")
                src_toks_decomp = [split_word(tok, splits=compound_splits) for tok in src_toks]
                src = " ".join(src_toks_decomp)

                #realign src with SIM
                aligned_SIM = aligner_SIM.get_word_aligns(src, trg)
                aligned_str = ""
                for src_align_id, trg_align_id in aligned_SIM["fwd"]:
                    aligned_str += str(src_align_id) + "-" + str(trg_align_id) + " "
                aligned_str_SIM = aligned_str.strip()

                #postag src
                src_doc = SRC_NLP(spacy.tokens.Doc(SRC_NLP.vocab, words=src.split(" ")))
                src_pos = [tok.pos_ for tok in src_doc]  # (tok.text, tok.pos_)
                src_pos_str = " ".join(src_pos)

                # calculate nulls

                #write to file
                g.write("\n" + sent_id + "\t" + src + "\t" + trg + "\t" + src_pos_str + "\t" + pos_trg + "\t" + "\t" +aligned_str_SIM+ "\t" + "\t" + "\t" + "\t" + "\t" + "\t" +ent_src+ "\t"+ent_trg)

                count += 1



if __name__ == "__main__":
    spacy.require_gpu()

    corp_name = sys.argv[1]
    print("=" * 80)
    print("DECOMPOSE " + str(corp_name))
    print("=" * 80)

    NLP_SRC, NLP_TRG = choose_model(corp_name)

    if "2de" in corp_name:
        decompose_DE_trg(CORPUS_DIR+corp_name+"_extracted.txt", CORPUS_DIR+corp_name+"_decomposed.txt", NLP_SRC, NLP_TRG)
    elif "de2" in corp_name:
        decompose_DE_src(CORPUS_DIR+corp_name+"_extracted.txt", CORPUS_DIR+corp_name+"_decomposed.txt", NLP_SRC, NLP_TRG)
    else: print("ERROR: no German file")