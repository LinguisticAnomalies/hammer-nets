'''
calculate log lexical frequency on generated text
'''
import string
from datetime import datetime
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from scipy.stats import ttest_ind
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import accumu_model_driver, generate_texts
from util_fun import get_word_lf, load_word_dist
from util_fun import break_attn_heads_by_layer



def pre_process(res_df):
    """
    add more stopwords a list of generated text,
    remove stopwords from the generate text
    return the cleaned tokens

    :param res_df: the dataframe for langauge generation
    :type res_df: pd.DataFrame
    :return: control model tokens and dementia model tokens
    :rtype: list, list
    """
    con_text = res_df["control"].values.tolist()
    dem_text = res_df["dementia"].values.tolist()
    bird_text = res_df["sentence"].values.tolist()
    # pre-process
    con_text = " ".join(con_text).lower()
    dem_text = " ".join(dem_text).lower()
    bird_text = " ".join(bird_text).lower()
    stop_words = stopwords.words("english")
    con_tokens = word_tokenize(con_text)
    dem_tokens = word_tokenize(dem_text)
    bird_tokens = word_tokenize(bird_text)
    # remove punctuation
    con_tokens = [token for token in con_tokens if token not in string.punctuation]
    dem_tokens = [token for token in dem_tokens if token not in string.punctuation]
    bird_tokens = [token for token in bird_tokens if token not in string.punctuation]
    # add more stop words
    stop_words.append("n't")
    # add words starting with '
    con_temp = [token for token in con_tokens if token.startswith("'")]
    stop_words.extend(con_temp)
    dem_temp = [token for token in dem_tokens if token.startswith("'")]
    stop_words.extend(dem_temp)
    tagged = pos_tag(con_tokens)
    for item in tagged:
        tag = item[1]
        token = item[0]
        if tag in ("PRP", "PRP$", "WP$", "EX"):
            stop_words.append(token)
    tagged = pos_tag(dem_tokens)
    for item in tagged:
        tag = item[1]
        token = item[0]
        if tag in ("PRP", "PRP$", "WP$", "EX"):
            stop_words.append(token)
    con_tokens = [token for token in con_tokens if token not in stop_words]
    dem_tokens = [token for token in dem_tokens if token not in stop_words]
    return con_tokens, dem_tokens


def calculate_lexical_frequency(con_tokens, dem_tokens):
    """
    calculate the lexical frequency for control & dementia model output,
    run t-test on two list

    :param con_tokens: a list of pre-processed control model output tokens
    :type con_tokens: list
    :param dem_tokens: a list of pre-processed dementia model output tokens
    :type dem_tokens: list
    """
    word_dist = load_word_dist()
    con_lf = [get_word_lf(token, word_dist) for token in con_tokens]
    con_lf = [token for token in con_lf if token != -np.inf]
    dem_lf = [get_word_lf(token, word_dist) for token in dem_tokens]
    dem_lf = [token for token in dem_lf if token != -np.inf]
    print("control model log lexical frequency: {}".format(round(sum(con_lf)/len(con_lf), 2)))
    print("dementia model log lexical frequency: {}".format(round(sum(dem_lf)/len(dem_lf), 2)))
    print("control model unique word ratio: {:0.2f}".format(len(set(con_tokens))/len(con_tokens)))
    print("dementia model unique word ratio: {:0.2f}".format(len(set(dem_tokens))/len(dem_tokens)))
    # if p<0.05 -> significant difference between two samples
    print("t-test p-value: {:0.3f}".format(ttest_ind(con_lf, dem_lf, alternative="less")[1]))


def cal_driver(data_name):
    """
    calculate the lexical frequency for generated text,
    the driver function

    :param data_name: the best pattern given the data name
    """
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    zero_style = "first"
    bird_df = pd.read_csv("data/bird_frame.tsv", sep="\t")
    bird_all = bird_df[bird_df["file"] == "mct_all.txt"]["text"].values.tolist()[0]
    bird_sents = sent_tokenize(bird_all)
    if data_name == "adr":
        share = 50
        layers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    elif data_name == "db":
        share = 50
        layers = [0, 1, 2, 3, 4, 5]
    elif data_name == "ccc":
        share = 50
        layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    else:
        raise ValueError("wrong data name")
    for layer in layers:
        model_dem = break_attn_heads_by_layer(zero_style, model_dem, 
                                              share, layer)
    lan_gene = generate_texts(model_con, model_dem,
                              gpt_tokenizer, bird_sents)
    con_tokens, dem_tokens = pre_process(lan_gene)
    calculate_lexical_frequency(con_tokens, dem_tokens)


if __name__ == "__main__":
    start_time = datetime.now()
    cal_driver("ccc")
    print("Total time running :{}\n".format(datetime.now() - start_time))
