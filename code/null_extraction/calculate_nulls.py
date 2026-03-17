import sys
import spacy

CORPUS_DIR = "../NEW_corpus/"

PUNCTUATION = [".", ",","-",";","?","!","&quot;"]


def calculate_nulls_SIM_decomp(readfile_decomp, writefile):
    with open(readfile_decomp) as f:
        with open(writefile, "w") as g:
            firstline = True
            data = f.read()
            data_list = data.split("\n")

            for count in range(len(data_list)):
                if firstline:
                    g.write(data_list[count])
                    count += 1
                    firstline = False
                    continue
                line_l = data_list[count].split("\t")

                sent_id = line_l[0]
                src = line_l[1]
                trg = line_l[2]
                src_tokked = src.split(" ")
                trg_tokked = trg.split(" ")
                src_pos = line_l[3]
                trg_pos = line_l[4]
                align_str_sim = line_l[6]
                align_list_sim = align_str_sim.split(" ")
                align_sim = [elem.split("-") for elem in align_list_sim]
                ent_src = line_l[12]
                ent_trg = line_l[13]


                if count % 1000 == 0: print(count)

                # calculate nulls

                # if no word is aligned, set aligenment list to emptylist
                source_alignment_ids_sim = list()
                target_alignment_ids_sim = list()
                if align_str_sim.strip() != "":
                    for elem in align_list_sim:
                        elem_split = elem.split("-")
                        source_alignment_ids_sim.append(elem_split[0])
                        target_alignment_ids_sim.append(elem_split[1])


                # compute null alignments

                explicitation_ids_sim = list(
                    filter(lambda i: str(i) not in target_alignment_ids_sim and trg[i] not in PUNCTUATION,
                           [i for i in
                            range(
                                len(trg_tokked))]))  # list of indexes in translation that do not have a corresponding original word (explicitation)

                simplification_ids_sim = list(
                    filter(lambda i: str(i) not in source_alignment_ids_sim and src[i] not in PUNCTUATION,
                           [i for i in
                            range(
                                len(src_tokked))]))  # list of indexes in original that do not have a corresponding translated words (simplification)

                explicitation_words_and_ids_sim = [(trg_tokked[i], i) for i in explicitation_ids_sim]
                simplification_words_and_ids_sim = [(src_tokked[i], i) for i in simplification_ids_sim]


                # write line to file
                expl_items_sim = ""
                for word, id in explicitation_words_and_ids_sim:
                    item_str = word + "," + str(id) + " "
                    expl_items_sim += item_str
                expl_items_sim = expl_items_sim.strip()

                simpl_items_sim = ""
                for word, id in simplification_words_and_ids_sim:
                    item_str = word + "," + str(id) + " "
                    simpl_items_sim += item_str
                simpl_items_sim = simpl_items_sim.strip()

                # write to file
                g.write("\n" + sent_id + "\t" + src + "\t" + trg + "\t" + src_pos + "\t" + trg_pos + "\t" + "\t" +align_str_sim+ "\t" + "\t" +expl_items_sim+ "\t" + "\t" +simpl_items_sim+ "\t" + "\t" + ent_src + "\t" + ent_trg)

                count += 1


def calculate_nulls(readfile_pos, readfile_align, writefile):

    with open(readfile_align) as f_align:
        with open(readfile_pos) as f_pos:
            with open(writefile, "w") as g:
                firstline = True
                data_pos = f_pos.read()
                data_pos_list = data_pos.split("\n")
                data_align = f_align.read()
                data_align_list = data_align.split("\n")

                for count in range(len(data_pos_list)):
                    if firstline:
                        g.write(data_pos_list[count])
                        count += 1
                        firstline = False
                        continue
                    line_l_pos = data_pos_list[count].split("\t")
                    line_l_align = data_align_list[count].split("\t")

                    sent_id = line_l_pos[0]
                    src = line_l_pos[1]
                    trg = line_l_pos[2]
                    src_tokked = src.split(" ")
                    trg_tokked = trg.split(" ")
                    src_pos = line_l_pos[3]
                    trg_pos = line_l_pos[4]
                    align_str_efl = line_l_align[5]
                    align_str_sim = line_l_align[6]
                    align_list_efl = align_str_efl.split(" ")
                    align_list_sim = align_str_sim.split(" ")
                    align_efl = [elem.split("-") for elem in align_list_efl]
                    align_sim = [elem.split("-") for elem in align_list_sim]


                    if count % 1000 == 0: print(count)

                    # calculate nulls

                    # if no word is aligned, set aligenment list to emptylist
                    source_alignment_ids_efl = list()
                    target_alignment_ids_efl = list()
                    if align_str_efl.strip() != "":
                        for elem in align_list_efl:
                            elem_split = elem.split("-")
                            source_alignment_ids_efl.append(elem_split[0])
                            target_alignment_ids_efl.append(elem_split[1])

                    source_alignment_ids_sim = list()
                    target_alignment_ids_sim = list()
                    if align_str_sim.strip() != "":
                        for elem in align_list_sim:
                            elem_split = elem.split("-")
                            source_alignment_ids_sim.append(elem_split[0])
                            target_alignment_ids_sim.append(elem_split[1])


                    # compute null alignments
                    explicitation_ids_efl = list(
                        filter(lambda i: str(i) not in target_alignment_ids_efl and trg[i] not in PUNCTUATION,
                               [i for i in
                                range(
                                    len(trg_tokked))]))  # list of indexes in translation that do not have a corresponding original word (explicitation)

                    simplification_ids_efl = list(
                        filter(lambda i: str(i) not in source_alignment_ids_efl and src[i] not in PUNCTUATION,
                               [i for i in
                                range(
                                    len(src_tokked))]))  # list of indexes in original that do not have a corresponding translated words (simplification)

                    explicitation_words_and_ids_efl = [(trg_tokked[i], i) for i in explicitation_ids_efl]
                    simplification_words_and_ids_efl = [(src_tokked[i], i) for i in simplification_ids_efl]


                    explicitation_ids_sim = list(
                        filter(lambda i: str(i) not in target_alignment_ids_sim and trg[i] not in PUNCTUATION,
                               [i for i in
                                range(
                                    len(trg_tokked))]))  # list of indexes in translation that do not have a corresponding original word (explicitation)

                    simplification_ids_sim = list(
                        filter(lambda i: str(i) not in source_alignment_ids_sim and src[i] not in PUNCTUATION,
                               [i for i in
                                range(
                                    len(src_tokked))]))  # list of indexes in original that do not have a corresponding translated words (simplification)

                    explicitation_words_and_ids_sim = [(trg_tokked[i], i) for i in explicitation_ids_sim]
                    simplification_words_and_ids_sim = [(src_tokked[i], i) for i in simplification_ids_sim]


                    # write line to file

                    expl_items_efl = ""
                    for word, id in explicitation_words_and_ids_efl:
                        item_str = word + "," + str(id) + " "
                        expl_items_efl += item_str
                    expl_items_efl = expl_items_efl.strip()

                    simpl_items_efl = ""
                    for word, id in simplification_words_and_ids_efl:
                        item_str = word + "," + str(id) + " "
                        simpl_items_efl += item_str
                    simpl_items_efl = simpl_items_efl.strip()


                    expl_items_sim = ""
                    for word, id in explicitation_words_and_ids_sim:
                        item_str = word + "," + str(id) + " "
                        expl_items_sim += item_str
                    expl_items_sim = expl_items_sim.strip()

                    simpl_items_sim = ""
                    for word, id in simplification_words_and_ids_sim:
                        item_str = word + "," + str(id) + " "
                        simpl_items_sim += item_str
                    simpl_items_sim = simpl_items_sim.strip()

                    # write to file
                    g.write("\n" + sent_id + "\t" + src + "\t" + trg + "\t" + src_pos + "\t" + trg_pos + "\t" +align_str_efl+ "\t" +align_str_sim+ "\t" +expl_items_efl+ "\t" +expl_items_sim+ "\t" +simpl_items_efl+ "\t" +simpl_items_sim+ "\t" + "\t" + "\t")

                    count += 1

if __name__ == "__main__":
    spacy.require_gpu()

    corp_name = sys.argv[1]
    print("=" * 80)
    print("POS-TAGGING " + str(corp_name))
    print("=" * 80)

    if len(sys.argv) == 3:
        if sys.argv[2] == "decomp":
            calculate_nulls_SIM_decomp(CORPUS_DIR+corp_name+"_decomposed.txt", CORPUS_DIR+corp_name+"_decomposed_nulls.txt")
        else: print("2nd argument unknown")
    else:
        calculate_nulls(CORPUS_DIR+corp_name+"_pos.txt", CORPUS_DIR+corp_name+"_align.txt", CORPUS_DIR+corp_name+"_nulls.txt")