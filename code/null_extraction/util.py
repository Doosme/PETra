import spacy

def choose_model(corp_name):
    if "en2" in corp_name:
        NLP_SRC = spacy.load("en_core_web_trf")
    elif "de2" in corp_name:
        NLP_SRC = spacy.load("de_core_news_lg")
    elif "fr2" in corp_name:
        NLP_SRC = spacy.load("fr_core_news_lg")
    elif "es2" in corp_name:
        NLP_SRC = spacy.load("es_core_news_lg")
    elif "it2" in corp_name:
        NLP_SRC = spacy.load("it_core_news_lg")
    elif "pt2" in corp_name:
        NLP_SRC = spacy.load("pt_core_news_lg")
    elif "ro2" in corp_name:
        NLP_SRC = spacy.load("ro_core_news_lg")
    elif "nb2" in corp_name:
        NLP_SRC = spacy.load("nb_core_news_lg")
    elif "sv2" in corp_name:
        NLP_SRC = spacy.load("sv_core_news_lg")
    elif "nl2" in corp_name:
        NLP_SRC = spacy.load("nl_core_news_lg")
    elif "ca2" in corp_name:
        NLP_SRC = spacy.load("ca_core_news_lg")
    elif "el2" in corp_name:
        NLP_SRC = spacy.load("el_core_news_lg")
    else:
        print("ERROR: SRC language not found!")

    if "2en" in corp_name:
        NLP_TRG = spacy.load("en_core_web_trf")
    elif "2de" in corp_name:
        NLP_TRG = spacy.load("de_core_news_lg")
    elif "2fr" in corp_name:
        NLP_TRG = spacy.load("fr_core_news_lg")
    elif "2es" in corp_name:
        NLP_TRG = spacy.load("es_core_news_lg")
    elif "2it" in corp_name:
        NLP_TRG = spacy.load("it_core_news_lg")
    elif "2pt" in corp_name:
        NLP_TRG = spacy.load("pt_core_news_lg")
    elif "2ro" in corp_name:
        NLP_TRG = spacy.load("ro_core_news_lg")
    elif "2nb" in corp_name:
        NLP_TRG = spacy.load("nb_core_news_lg")
    elif "2sv" in corp_name:
        NLP_TRG = spacy.load("sv_core_news_lg")
    elif "2nl" in corp_name:
        NLP_TRG = spacy.load("nl_core_news_lg")
    elif "2ca" in corp_name:
        NLP_TRG = spacy.load("ca_core_news_lg")
    elif "2el" in corp_name:
        NLP_TRG = spacy.load("el_core_news_lg")
    else:
        print("ERROR: TRG language not found!")

    return NLP_SRC, NLP_TRG



def choose_model_TRG(corp_name):
    if "2en" in corp_name:
        NLP_TRG = spacy.load("en_core_web_trf")
    elif "2de" in corp_name:
        NLP_TRG = spacy.load("de_core_news_lg")
    elif "2fr" in corp_name:
        NLP_TRG = spacy.load("fr_core_news_lg")
    elif "2es" in corp_name:
        NLP_TRG = spacy.load("es_core_news_lg")
    elif "2it" in corp_name:
        NLP_TRG = spacy.load("it_core_news_lg")
    elif "2pt" in corp_name:
        NLP_TRG = spacy.load("pt_core_news_lg")
    elif "2ro" in corp_name:
        NLP_TRG = spacy.load("ro_core_news_lg")
    elif "2nb" in corp_name:
        NLP_TRG = spacy.load("nb_core_news_lg")
    elif "2sv" in corp_name:
        NLP_TRG = spacy.load("sv_core_news_lg")
    elif "2nl" in corp_name:
        NLP_TRG = spacy.load("nl_core_news_lg")
    elif "2ca" in corp_name:
        NLP_TRG = spacy.load("ca_core_news_lg")
    elif "2el" in corp_name:
        NLP_TRG = spacy.load("el_core_news_lg")
    else:
        print("ERROR: TRG language not found!")

    return NLP_TRG


def choose_model_SRC(corp_name):
    if "en2" in corp_name:
        NLP_SRC = spacy.load("en_core_web_trf")
    elif "de2" in corp_name:
        NLP_SRC = spacy.load("de_core_news_lg")
    elif "fr2" in corp_name:
        NLP_SRC = spacy.load("fr_core_news_lg")
    elif "es2" in corp_name:
        NLP_SRC = spacy.load("es_core_news_lg")
    elif "it2" in corp_name:
        NLP_SRC = spacy.load("it_core_news_lg")
    elif "pt2" in corp_name:
        NLP_SRC = spacy.load("pt_core_news_lg")
    elif "ro2" in corp_name:
        NLP_SRC = spacy.load("ro_core_news_lg")
    elif "nb2" in corp_name:
        NLP_SRC = spacy.load("nb_core_news_lg")
    elif "sv2" in corp_name:
        NLP_SRC = spacy.load("sv_core_news_lg")
    elif "nl2" in corp_name:
        NLP_SRC = spacy.load("nl_core_news_lg")
    elif "ca2" in corp_name:
        NLP_SRC = spacy.load("ca_core_news_lg")
    elif "el2" in corp_name:
        NLP_SRC = spacy.load("el_core_news_lg")
    else:
        print("ERROR: SRC language not found!")
    return NLP_SRC