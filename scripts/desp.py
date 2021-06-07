'''
ADReSS and CCC dataset descriptive stats
'''

import pandas as pd
from nltk.tokenize import word_tokenize


adr_train = pd.read_csv("data/adress_train_full.tsv", sep="\t")
adr_train["set"] = "train"
adr_test = pd.read_csv("data/adress_test_full.tsv", sep="\t")
adr_test["set"] = "test"
adr_full = adr_train.append(adr_test)
adr_full["len"] = adr_full["text"].apply(word_tokenize).tolist()
adr_full["len"] = adr_full["len"].apply(len)



print("="*20)
print("CCC")
ccc = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
ccc["len"] = ccc["text"].apply(word_tokenize).tolist()
ccc["len"] = ccc["len"].apply(len)
print(ccc.groupby("label")["file"].nunique().reset_index())
print(ccc.groupby(["label"])["len"].mean().reset_index())
print(ccc.groupby(["label"])["len"].std().reset_index())



print("="*20)
print("full ADReSS by training/test set")
print(adr_full.groupby(["label", "set"])["file"].count().reset_index())
print(adr_full.groupby(["label", "set"])["mmse"].mean().reset_index())
print(adr_full.groupby(["label", "set"])["mmse"].std().reset_index())
print(adr_full.groupby(["label", "set"])["len"].mean().reset_index())
print(adr_full.groupby(["label", "set"])["len"].std().reset_index())

print("="*20)
print("full ADReSS")
print(adr_full.groupby(["label"])["file"].count().reset_index())
print(adr_full.groupby(["label"])["mmse"].mean().reset_index())
print(adr_full.groupby(["label"])["mmse"].std().reset_index())
print(adr_full.groupby(["label"])["len"].mean().reset_index())
print(adr_full.groupby(["label"])["len"].std().reset_index())

print("="*20)
print("DB")
db = pd.read_csv("data/db.tsv", sep="\t")
db["len"] = db["text"].apply(word_tokenize).tolist()
db["len"] = db["text"].apply(len)
print("Controls: ", len(set(db[db['label'] == 0]['file'])))
print("Dementia: ", len(set(db[db['label'] == 1]['file'])))
print(db.groupby(["label"])["file"].nunique().reset_index())
print(db.groupby(["label"])["mmse"].mean().reset_index())
print(db.groupby(["label"])["mmse"].std().reset_index())
print(db.groupby(["label"])["len"].mean().reset_index())
print(db.groupby(["label"])["len"].std().reset_index())