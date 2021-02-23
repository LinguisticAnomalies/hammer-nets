'''
the script for dataset ground truth stats
'''


import pickle
import pandas as pd
import numpy as np


def get_data(data_type):
    """
    get dataset by data type

    :param data_type: one of four data perspectives, including
                      'full', 'mild', 'slight', 'sev' and 'ccc'
    :type data_type: str
    """
    if data_type == "full":
        train_df = pd.read_csv("data/address_train_full.tsv", sep="\t")
        test_df = pd.read_csv("data/address_test_full.tsv", sep="\t")
    elif data_type == "mild":
        train_df = pd.read_csv("data/address_train_mild.tsv", sep="\t")
        test_df = pd.read_csv("data/address_test_mild.tsv", sep="\t")
    elif data_type == "slight":
        train_df = pd.read_csv("data/address_train_slight.tsv", sep="\t")
        test_df = pd.read_csv("data/address_test_slight.tsv", sep="\t")
    elif data_type == "sev":
        train_df = pd.read_csv("data/address_train_sev.tsv", sep="\t")
        test_df = pd.read_csv("data/address_test_sev.tsv", sep="\t")
    elif data_type == "ccc":
        train_df = pd.read_csv("data/ccc_train.tsv", sep="\t")
        test_df = pd.read_csv("data/ccc_test.tsv", sep="\t")
    else:
        raise ValueError("data type is not supported!")
    return train_df, test_df


def get_desp(data_type):
    """
    get descriptive stats with data type

    :param data_type: one of four data perspectives, including
                      'full', 'mild', 'slight', 'sev' and 'ccc'
    :type data_type: str
    """
    train_df, test_df = get_data(data_type)
    print("="*20)
    print("{} training set".format(data_type))
    print(train_df.groupby("label").describe().unstack(1))
    print("="*20)
    print("{} test set".format(data_type))
    print(test_df.groupby("label").describe().unstack(1))


def load_res_dict(file_path):
    """
    load the result dict from the pickle file,
    return the dict

    :param file_path: the path to the pickle file
    :type file_path: str
    :return: the dict with c-d, c/d evaluation results
    :rtype: dict
    """
    res_dict = {"train_diff_auc": [], "train_diff_accu": [],
                "test_diff_auc": [], "test_diff_accu": [],
                "train_ratio_auc": [], "train_ratio_accu": [],
                "test_ratio_auc": [], "test_ratio_accu": []}
    with open (file_path, "rb") as f:
        file_dict = pickle.load(f)
        for key in res_dict.keys():
            res_dict[key].extend(file_dict[key])
    return res_dict


def compare_hammer_eval_metrics(share, zero_style, data_type, model_type):
    """
    compare an evaluation metrics with given share and zero on different hammer sytles
    for this function, comparing 'onetime', 'accumu' and 'comb'

    :param share: % attention heads zeroed
    :type share: int
    :param zero_style: either 'first' or 'random'
    :type zero_style: str
    :param data_type: data subset style
    :type data_type: str
    :param model_type: either c/d or c-d models
    :type: str
    """
    # comb
    comb_file = "../results/evals/comb_{}_{}_{}.pkl".format(zero_style, share, data_type)
    comb_dict = load_res_dict(comb_file)
    # onetime
    onetime_file = "../results/evals/onetime_{}_{}_{}.pkl".format(zero_style, share, data_type)
    onetime_dict = load_res_dict(onetime_file)
    # accumu
    accumu_file = "../results/evals/accumu_{}_{}_{}.pkl".format(zero_style, share, data_type)
    accumu_dict = load_res_dict(accumu_file)
    print("="*20)
    train_auc = "train_{}_auc".format(model_type)
    train_accu = "train_{}_accu".format(model_type)
    test_auc = "test_{}_auc".format(model_type)
    test_accu = "test_{}_accu".format(model_type)
    print("{}% {} training {} auc & accuracy".format(share, data_type, model_type))
    print("comb   : {:0.2f}\t{:0.2f}".format(comb_dict[train_auc][0],
                                             comb_dict[train_accu][0]))
    print("accumu : {:0.2f}\t{:0.2f}".format(np.mean(accumu_dict[train_auc]),
                                             np.mean(accumu_dict[train_accu])))
    print("onetime: {:0.2f}\t{:0.2f}".format(np.mean(onetime_dict[train_auc]),
                                             np.mean(onetime_dict[train_accu])))
    print("="*20)
    print("{}\% {} test {} auc & accuracy".format(share, data_type, model_type))
    print("comb   : {:0.2f}\t{:0.2f}".format(comb_dict[test_auc][0],
                                             comb_dict[test_accu][0]))
    print("accumu : {:0.2f}\t{:0.2f}".format(np.mean(accumu_dict[test_auc]),
                                             np.mean(accumu_dict[test_accu])))
    print("onetime: {:0.2f}\t{:0.2f}".format(np.mean(onetime_dict[test_auc]),
                                             np.mean(onetime_dict[test_accu])))


if __name__ == "__main__":
    #compare_hammer_eval_metrics(100, "random", "full", "ratio")
    df = pd.read_csv("data/ccc_test.tsv", sep="\t")
    print(df["text"].values.tolist()[0])
    print(type(print(df["text"].values.tolist()[0])))