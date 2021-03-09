'''
ADReSS and CCC dataset descriptive stats
'''

import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.stats import ttest_ind


adr_train = pd.read_csv("data/address_train_full.tsv", sep="\t")
adr_train["set"] = 0
adr_test = pd.read_csv("data/address_test_full.tsv", sep="\t")
adr_test["set"] = 1
adr_full = adr_train.append(adr_test)
adr_full["len"] = adr_full["text"].apply(word_tokenize).tolist()
adr_full["len"] = adr_full["len"].apply(len)
print("ADReSS dataset transcript lenth")
print(adr_full.shape)
print(adr_full.groupby(["label", "set"])["len"].mean().reset_index())
# t-test

# train non-ad vs. ad
adr_train_nonad = adr_full.loc[(adr_full["set"] == 0) & adr_full["label"] == 0]["len"].values
adr_train_ad = adr_full.loc[(adr_full["set"] == 0) & adr_full["label"] == 1]["len"].values
adr_test_nonad = adr_full.loc[(adr_full["set"] == 1) & adr_full["label"] == 0]["len"].values
adr_test_ad = adr_full.loc[(adr_full["set"] == 1) & adr_full["label"] == 1]["len"].values

# p-values
print("t-test p-value for training set: {:0.3f}".format(ttest_ind(adr_train_ad, adr_train_nonad)[1]))
print("t-test p-value for test set: {:0.3f}".format(ttest_ind(adr_test_ad, adr_test_nonad)[1]))
print("t-test p-value for non-ad transcript: {:0.3f}".format(ttest_ind(adr_train_nonad, adr_test_nonad)[1]))
print("t-test p-value for ad transcript: {:0.3f}".format(ttest_ind(adr_train_ad, adr_test_ad)[1]))
print("="*20)
ccc = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
ccc["len"] = ccc["text"].apply(word_tokenize).tolist()
ccc["len"] = ccc["len"].apply(len)
print("CCC dataset transcript lenth")
print(ccc.groupby("label")["len"].mean().reset_index())
print("ccc dataset size: {}".format(ccc.shape))
print("ccc unique pid: {}".format(ccc["file"].unique().shape[0]))
ccc_count = ccc.groupby(["file", "label"])["text"].count().reset_index().sort_values("text", ascending=False)
print("pid with more than 1 transcript: {}".format(ccc_count[ccc_count["text"] > 1].shape[0]))
print("non ad pid with more than 1 transcript: {}".format(ccc_count.loc[(ccc_count["text"] > 1) & (ccc_count["label"] == 0)].shape[0]))
print("ad pid with more than 1 transcript: {}".format(ccc_count.loc[(ccc_count["text"] > 1) & (ccc_count["label"] == 1)].shape[0]))
print("CCC AD patients: {}".format(ccc[ccc["label"] == 1]["file"].unique().shape[0]))
print("CCC Non-AD patients: {}".format(ccc[ccc["label"] == 0]["file"].unique().shape[0]))
ccc_ad_len = ccc[ccc["label"] == 1]["len"].values
ccc_nonad_len = ccc[ccc["label"] == 0]["len"].values
print("t-test p-value for ccc transcript: {:0.3f}".format(ttest_ind(ccc_ad_len, ccc_nonad_len)[1]))