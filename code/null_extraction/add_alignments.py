CORPUS_DIR = "../NEW_corpus/"

import sys
import eflomal
import io
import os
import tempfile

import simalign


PUNCTUATION = [".", ",","-",";","?","!", "؟", "،", "؛"]


def _read_in_corpus_file_EFL(filename):
    src_sents_lst = list()
    trg_sents_lst = list()
    firstline = True

    with open(filename) as f:
        data = f.read()
        data_list = data.split("\n")
        for line in data_list:
            if firstline:
                firstline = False
                continue
            line_l = line.split("\t")
            src_sents_lst.append(line_l[1])
            trg_sents_lst.append(line_l[2])

    return src_sents_lst, trg_sents_lst

def _align_sentences_EFL(src_sents_lst, trg_sents_lst):
    aligner = eflomal.Aligner()
    # The aligner expects the source and target sentences in files with one
    # sentence per line. We use streams instead of "real" files to save the
    # additional I/O of writing the data to files first.
    stream_en = io.StringIO("\n".join(src_sents_lst) + "\n")
    stream_trg = io.StringIO("\n".join(trg_sents_lst) + "\n")

    # The aligner writes the links into a specified file, and there is no way to
    # avoid going to disk. We create a temporary file in the OS's default place for
    # temporary files (its name begins with "links_" and ends with ".fwd" if you
    # need to clean it up manually), let the aligner write the links to it, read
    # the links, and then remove the file.
    links_fwd_file_handle, links_fwd_file_path = tempfile.mkstemp(prefix='links_', suffix='.fwd')
    os.close(links_fwd_file_handle)

    # Run the aligner.
    aligner.align(stream_en, stream_trg, links_filename_fwd=links_fwd_file_path)

    # Read the links.
    with open(links_fwd_file_path) as f:
        links_fwd = f.read()[:-1].split("\n")

    # Remove the temporary file.
    os.remove(links_fwd_file_path)


    return links_fwd



def add_alignments(readfile, writefile):
    #align sentences with EFLOMAL
    print("start EFLOMAL")
    src_sents_lst, trg_sents_lst = _read_in_corpus_file_EFL(readfile)
    links_fwd_EFL = _align_sentences_EFL(src_sents_lst, trg_sents_lst)

    aligner_SIM = simalign.SentenceAligner(matching_methods="f") #f = forward; m = match
    print("done EFLOMAL")


    #iterate and aligne with SIMALIGN
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
                sent_id_tag = line_l[0]
                src = line_l[1]
                trg = line_l[2]
                src_pos = line_l[3]
                trg_pos = line_l[4]

                if count % 1000 == 0: print("simalign: " + str(count))


                #eflomal_alignment
                aligned_EFL = links_fwd_EFL[count]

                #sim_alignment
                aligned_SIM = aligner_SIM.get_word_aligns(src, trg)
                aligned_str = ""
                for src_align_id, trg_align_id in aligned_SIM["fwd"]:
                    aligned_str += str(src_align_id) + "-" + str(trg_align_id) + " "
                aligned_str_SIM = aligned_str.strip()


                # write to file
                g.write("\n" + sent_id_tag + "\t" + src + "\t" + trg + "\t" + src_pos + "\t" + trg_pos + "\t" +aligned_EFL+ "\t" +aligned_str_SIM+ "\t" + "\t" + "\t" + "\t" + "\t" + "\t" + "\t")

                count += 1



if __name__ == "__main__":
    corp_name = sys.argv[1]
    print("=" * 80)
    print("ALIGNING " + str(corp_name))
    print("=" * 80)

    add_alignments(CORPUS_DIR + corp_name + ".txt", CORPUS_DIR + corp_name + "_align.txt")