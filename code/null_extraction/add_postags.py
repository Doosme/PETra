import sys
import spacy


CORPUS_DIR = "../NEW_corpus/"

from util import choose_model

def add_pos_tags(readfile, writefile, SRC_NLP, TRG_NLP):
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

                if count % 1000 == 0: print(count)

                #create pos-tags
                src_doc = SRC_NLP(spacy.tokens.Doc(SRC_NLP.vocab, words=src.split(" ")))
                src_pos = [tok.pos_ for tok in src_doc]  #(tok.text, tok.pos_)
                src_pos_str = " ".join(src_pos)

                trg_doc = TRG_NLP(spacy.tokens.Doc(TRG_NLP.vocab, words=trg.split(" ")))
                trg_pos = [tok.pos_ for tok in trg_doc]  #(tok.text, tok.pos_)
                trg_pos_str = " ".join(trg_pos)

                #write to file
                g.write("\n" + sent_id + "\t" + src + "\t" + trg + "\t" + src_pos_str + "\t" + trg_pos_str + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t")

                count += 1

if __name__ == "__main__":
    spacy.require_gpu()

    corp_name = sys.argv[1]
    print("=" * 80)
    print("POS-TAGGING " + str(corp_name))
    print("=" * 80)

    NLP_SRC, NLP_TRG = choose_model(corp_name)

    add_pos_tags(CORPUS_DIR+corp_name+".txt", CORPUS_DIR+corp_name+"_pos.txt", NLP_SRC, NLP_TRG)