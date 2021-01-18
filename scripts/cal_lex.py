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


def add_more_stopwords(stop_words, tokens):
    """
    add more stopwrods a list of generated text
    return the new stop word list

    :param stop_words: a list of stop words
    :type stop_words: list
    :param tokens: a list of tokens from generated text
    :type sents: list
    :return: the updated stop word list
    :rtype: list
    """
    tagged = pos_tag(tokens)
    for item in tagged:
        tag = item[1]
        token = item[0]
        if tag in ("PRP", "PRP$", "WP$", "EX"):
            stop_words.append(token)
    return stop_words


def calculate_lexcial_frequency(hammer_style, zero_style,
                                share, data_type):
    """
    calculate the lexical frequency for generated text

    :param hammer_style: impaired style
    :type hammer_style: str
    :param zero_style: zeroing attn head style
    :type zero_style: str
    :param share: % of attn head impaired
    :type share: int
    :param data_type: address mmse dataset type
    :type data_type: str
    """
    res_file = "../results/{}_{}_{}_{}.tsv".format(hammer_style, zero_style, share, data_type)
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
    stop_words = add_more_stopwords(stop_words, con_tokens)
    stop_words = add_more_stopwords(stop_words, dem_tokens)
    stop_words = add_more_stopwords(stop_words, bird_tokens)
    stop_words.append("n't")
    # add words starting with '
    con_temp = [token for token in con_tokens if token.startswith("'")]
    stop_words.extend(con_temp)
    dem_temp = [token for token in dem_tokens if token.startswith("'")]
    stop_words.extend(dem_temp)
    # calculate lexical frequency
    word_dist = load_word_dist()
    # remove stopwords
    con_tokens = [token for token in con_tokens if token not in stop_words]
    dem_tokens = [token for token in dem_tokens if token not in stop_words]
    # calculate lexical frequency
    con_lf = [get_word_lf(token, word_dist) for token in con_tokens]
    con_lf = [token for token in con_lf if token != -np.inf]
    dem_lf = [get_word_lf(token, word_dist) for token in dem_tokens]
    dem_lf = [token for token in dem_lf if token != -np.inf]
    print("="*20)
    print("control model log lexical frequency: {}".format(sum(con_lf)/len(con_lf)))
    print("dementia model log lexical frequency: {}".format(sum(dem_lf)/len(dem_lf)))
    print("control model unique word ratio: {}".format(len(set(con_tokens))/len(con_tokens)))
    print("dementia model unique word ratio: {}".format(len(set(dem_tokens))/len(dem_tokens)))
    print("t-test p-value: {}".format(ttest_ind(con_lf, dem_lf)[1]))
    con_common = [token for token, token_count in Counter(con_tokens).most_common(10)]
    dem_common = [token for token, token_count in Counter(dem_tokens).most_common(10)]
    print("top 10 most common tokens in control text: {}".format(con_common))
    print("top 10 most common tokens in dementia text: {}".format(dem_common))
    print("="*20)


if __name__ == "__main__":
    start_time = datetime.now()
    args = parse_args()
    calculate_lexcial_frequency(args.hammer_style, args.zero_style, args.share, args.data_type)
    print("Total time running :{}\n".format(datetime.now() - start_time))
