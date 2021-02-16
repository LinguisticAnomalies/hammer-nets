'''
the script for dataset ground truth stats
'''

import pickle
import pandas as pd
import numpy as np


def get_address_desp(data_type):
    """
    print out ADDReSS dataset descriptive statistics by given data type

    :param data_type: one of four data perspectives, including
                      'full', 'mild', 'slight' and 'sev'
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
    else:
        raise ValueError("data type is not supported!")
    print("="*20)
    print("{} training set".format(data_type))
    print(train_df.groupby("label").describe().unstack(1))
    print("="*20)
    print("{} test set".format(data_type))
    print(test_df.groupby("label").describe().unstack(1))


if __name__ == "__main__":
    # get_address_desp("mild")
    with open("/edata/dementia_cleaned.pkl", "rb") as f:
        df = pickle.load(f)
    df["label"] = np.where(df["dementia"] == True, 1, 0)
    trans = df["Transcript"].values.tolist()
    for i in range(5):
        print(trans[i])
        print()
    print(df.columns)