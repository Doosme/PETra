import sys
import spacy


CORPUS_DIR = "../NEW_corpus/"

from util import choose_model




def named_entity_recognition(readfile, writefile, SRC_NLP, TRG_NLP):
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
                ent_list_src = [str(ent.text) + "("+str(ent.label_)+")" for ent in src_doc.ents]
                entities_src = "; ".join(ent_list_src)
                #print(entities_src)

                trg_doc = TRG_NLP(spacy.tokens.Doc(TRG_NLP.vocab, words=trg.split(" ")))
                ent_list_trg = [str(ent.text) + "("+str(ent.label_)+")" for ent in trg_doc.ents]
                entities_trg = " ; ".join(ent_list_trg)
                #print(entities_trg)


                #write to file
                g.write("\n" + sent_id + "\t" + src + "\t" + trg + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t" +entities_src+ "\t"+entities_trg)

                count += 1


if __name__ == "__main__":
    spacy.require_gpu()

    corp_name = sys.argv[1]
    print("=" * 80)
    print("NAMED-ENTITY RECOGNITION " + str(corp_name))
    print("=" * 80)

    NLP_SRC, NLP_TRG = choose_model(corp_name)

    named_entity_recognition(CORPUS_DIR+corp_name+".txt", CORPUS_DIR+corp_name+"_ner.txt", NLP_SRC, NLP_TRG)