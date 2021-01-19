'''
calculate log lexical frequency on generated text
'''
import string
import argparse
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from scipy.stats import ttest_ind
from util_fun import get_word_lf, load_word_dist


def parse_args():
    """
    add arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str,
                        help="address mmse dataset type")
    parser.add_argument("--share", type=int,
                        help="the % of attn head impaired")
    parser.add_argument("--hammer_style", type=str,
                        help="impaired style")
    parser.add_argument("--zero_style", type=str,
                        help="zero attn heads style")
    return parser.parse_args()


def pre_process(res_file):
    """
    add more stopwords a list of generated text,
    remove stopwords from the generate text
    return the cleaned tokens

    :param res_file: the path to text generation dataset
    :type res_file: str
    :return: control model tokens and dementia model tokens
    :rtype: list, list
    """
    res_df = pd.read_csv(res_file, sep="\t")
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
    print("control model log lexical frequency: {}".format(sum(con_lf)/len(con_lf)))
    print("dementia model log lexical frequency: {}".format(sum(dem_lf)/len(dem_lf)))
    print("control model unique word ratio: {}".format(len(set(con_tokens))/len(con_tokens)))
    print("dementia model unique word ratio: {}".format(len(set(dem_tokens))/len(dem_tokens)))
    print("t-test p-value: {}".format(ttest_ind(con_lf, dem_lf)[1]))
    con_common = [token for token, token_count in Counter(con_tokens).most_common(10)]
    dem_common = [token for token, token_count in Counter(dem_tokens).most_common(10)]
    print("top 10 most common tokens in control text: {}".format(con_common))
    print("top 10 most common tokens in dementia text: {}".format(dem_common))


def cal_driver(hammer_style, zero_style, share, data_type):
    """
    calculate the lexical frequency for generated text,
    the driver function

    :param hammer_style: impaired style
    :type hammer_style: str
    :param zero_style: zeroing attn head style
    :type zero_style: str
    :param share: % of attn head impaired
    :type share: int
    :param data_type: address mmse dataset type
    :type data_type: str
    """
    if hammer_style == "comb":
        res_file = "../results/text/{}_{}_{}_{}.tsv".format(hammer_style, zero_style, share, data_type)
        con_tokens, dem_tokens = pre_process(res_file)
        print("=== {} model {} {}% log lexical frequency =====".format(hammer_style, zero_style, share))
        calculate_lexical_frequency(con_tokens, dem_tokens)
    elif hammer_style == "accumu":
        for layer in range(1, 13):
            print("=== {} model {} {}% log first {} layer lexical frequency =====".format(hammer_style, zero_style, share, layer))
            res_file = "../results/text/{}_{}_{}_{}_layer_{}.tsv".format(hammer_style, zero_style, share, data_type, layer)
            con_tokens, dem_tokens = pre_process(res_file)
            calculate_lexical_frequency(con_tokens, dem_tokens)
            print("=====================================")
    elif hammer_style == "onetime":
        for layer in range(0, 12):
            print("=== {} model {} {}% log {}-th layer lexical frequency =====".format(hammer_style, zero_style, share, layer))
            res_file = "../results/text/{}_{}_{}_{}_layer_{}.tsv".format(hammer_style, zero_style, share, data_type, layer)
            con_tokens, dem_tokens = pre_process(res_file)
            calculate_lexical_frequency(con_tokens, dem_tokens)
            print("=====================================")
    

if __name__ == "__main__":
    start_time = datetime.now()
    args = parse_args()
    cal_driver(args.hammer_style, args.zero_style, args.share, args.data_type)
    print("Total time running :{}\n".format(datetime.now() - start_time))
