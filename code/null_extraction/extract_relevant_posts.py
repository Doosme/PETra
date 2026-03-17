
import sys
import spacy


CORPUS_DIR = "../NEW_corpus/"

RELEVANT_ENT_TYPES = ["PERSON", "NORP", "FAC","ORG","GPE","LOC",
                      "PRODUCT", "EVENT", "WORK_OF_ART","LAW","LANGUAGE"]


def extract_SIM_decomp(decompnulls_file, writefile):
    with open(decompnulls_file) as f:
        with open(writefile, "w") as g:

            firstline = True
            data = f.read()
            data_list = data.split("\n")


            for count in range(len(data_list)):
                if firstline:
                    g.write(data_list[count].strip()+"\trelevant words")
                    firstline = False
                    continue
                line_l = data_list[count].split("\t")

                sent_id = line_l[0]
                src = line_l[1]
                trg = line_l[2]
                pos_src = line_l[3]
                pos_trg = line_l[4]
                align_sim = line_l[6]
                expl_sim = line_l[8]
                simpl_sim = line_l[10]
                ent_src = line_l[12]
                ent_trg = line_l[13]

                if count % 1000 == 0: print(count)

                if ent_trg.strip() == "":
                    print("SKIP1:" + sent_id)
                    continue #skip lines with no entity
                ent_trg_list = ent_trg.split(" ; ")
                new_ent_list = []
                for e in ent_trg_list:
                    e_l = e[0:-1].split("(")
                    ent_type = e_l[-1]
                    ent_word = e_l[1]
                    if ent_type in RELEVANT_ENT_TYPES:
                        new_ent_list.append(ent_word + "(" + ent_type + ")")
                if new_ent_list == list():
                    print("SKIP2:" + sent_id)
                    continue

                if expl_sim.strip() == "":
                    print("SKIP (empty): " + str(sent_id))
                    continue
                expl_sim_id_list = [int(item.split(",")[-1]) for item in expl_sim.split(" ")]
                pos_trg_list = pos_trg.split(" ")
                relevant_ids = list()
                for id in expl_sim_id_list:
                    if pos_trg_list[id] == "NOUN" or pos_trg_list[id] == "PROPN" or pos_trg_list[id] == "PRON":
                        relevant_ids.append(id)

                if relevant_ids == list():
                    print("SKIP2:" + sent_id)
                    continue

                #write to file
                g.write("\n" + sent_id + "\t" + src + "\t" + trg + "\t" + pos_src + "\t" + pos_trg + "\t" + "\t" +align_sim+ "\t" + "\t" +expl_sim+ "\t" + "\t" +simpl_sim+ "\t" + "\t" +ent_src+ "\t"+ent_trg)



def NEW__extract(ner_file, pos_file, align_file, nulls_file, writefile):
    with open(ner_file) as f_ner:
        with open(pos_file) as f_pos:
            with open(align_file) as f_align:
                with open(nulls_file) as f_nulls:
                    with open(writefile, "w") as g:

                        firstline = True
                        data_pos = f_pos.read()
                        data_ner = f_ner.read()
                        data_align = f_align.read()
                        data_nulls = f_nulls.read()
                        data_pos_list = data_pos.split("\n")
                        data_ner_list = data_ner.split("\n")
                        data_align_list = data_align.split("\n")
                        data_nulls_list = data_nulls.split("\n")


                        for count in range(len(data_pos_list)):
                            if firstline:
                                g.write(data_pos_list[count].strip()+"\trelevant words")
                                firstline = False
                                continue
                            line_l_pos = data_pos_list[count].split("\t")
                            line_l_ner = data_ner_list[count].split("\t")
                            line_l_align = data_align_list[count].split("\t")
                            line_l_nulls = data_nulls_list[count].split("\t")

                            sent_id = line_l_pos[0]
                            src = line_l_pos[1]
                            trg = line_l_pos[2]
                            pos_src = line_l_pos[3]
                            pos_trg = line_l_pos[4]
                            align_efl = line_l_align[5]
                            align_sim = line_l_align[6]
                            expl_efl = line_l_nulls[7]
                            expl_sim = line_l_nulls[8]
                            simpl_efl = line_l_nulls[9]
                            simpl_sim = line_l_nulls[10]
                            ent_src = line_l_ner[12]
                            ent_trg = line_l_ner[13]

                            if count % 1000 == 0: print(count)

                            if ent_trg.strip() == "":
                                print("SKIP1:" + sent_id)
                                continue #skip lines with no entity
                            ent_trg_list = ent_trg.split(" ; ")
                            new_ent_list = []
                            for e in ent_trg_list:
                                e_l = e[0:-1].split("(")
                                ent_type = e_l[-1]
                                ent_word = e_l[1]
                                if ent_type in RELEVANT_ENT_TYPES:
                                    new_ent_list.append(ent_word + "(" + ent_type + ")")
                            if new_ent_list == list():
                                print("SKIP2:" + sent_id)
                                continue

                            if expl_sim.strip() == "" or expl_efl.strip() == "":
                                print("SKIP (empty): " + str(sent_id))
                                continue
                            expl_efl_id_list = [int(item.split(",")[-1]) for item in expl_efl.split(" ")]
                            expl_sim_id_list = [int(item.split(",")[-1]) for item in expl_sim.split(" ")]
                            pos_trg_list = pos_trg.split(" ")
                            relevant_ids = list()
                            for id in list(set(expl_efl_id_list + expl_sim_id_list)):
                                if id in expl_efl_id_list and id in expl_sim_id_list:
                                    #if pos_trg_list[id] == "NOUN" or pos_trg_list[id] == "PROPN" or pos_trg_list[id] == "PRON": 
                                    if pos_trg_list[id] == "NOUN" or pos_trg_list[id] == "PROPN": 
                                        relevant_ids.append(id)

                            if relevant_ids == list():
                                print("SKIP2:" + sent_id)
                                continue

                            #write to file
                            g.write("\n" + sent_id + "\t" + src + "\t" + trg + "\t" + pos_src + "\t" + pos_trg + "\t" +align_efl+ "\t" +align_sim+ "\t" +expl_efl+ "\t" +expl_sim+ "\t" +simpl_efl+ "\t" +simpl_sim+ "\t" + "\t" +ent_src+ "\t"+ent_trg)





if __name__ == "__main__":
    spacy.require_gpu()

    corp_name = sys.argv[1]
    print("=" * 80)
    print("EXTRACTING " + str(corp_name))
    print("=" * 80)

    if len(sys.argv) == 3:
        if sys.argv[2] == "decomp":
            extract_SIM_decomp(CORPUS_DIR+corp_name+"_decomposed_nulls.txt", CORPUS_DIR+corp_name+"_decomposed_extracted.txt")
        else:
            print("2nd argument unknown")
    else:
        NEW__extract(CORPUS_DIR+corp_name+"_ner.txt", CORPUS_DIR+corp_name+"_pos.txt", CORPUS_DIR+corp_name+"_align.txt", CORPUS_DIR+corp_name+"_nulls.txt", CORPUS_DIR+corp_name+"_extracted.txt")
