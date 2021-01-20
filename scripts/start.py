"""
starter script
"""

import os
import pandas as pd
from util_fun import read_data, evaluate_model
from util_fun import process_wls_data
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PREFIX_CON = "/edata/lixx3013/dementia-data/DementiaBank//DemBank/Control/99/"
PREFIX_DEM = "/edata/lixx3013/dementia-data/DementiaBank/DemBank/Dementia/169/"
PREFIX_BIRD = "/edata/lixx3013/dementia-data/DementiaBank/DemBank/bird/"
PREFIX_CHILD_IMPAIRED = "/edata/lixx3013/dementia-data/Gillam/LongImpaired/txt/"
PREFIX_CHILD_NORMAL = "/edata/lixx3013/dementia-data/Gillam/Normal/txt/"
PERFIX_ADD_TRAIN_CON = "/edata/ADReSS-IS2020-data/transcription/train/cc/"
PERFIX_ADD_TRAIN_DEM = "/edata/ADReSS-IS2020-data/transcription/train/cd/"
PERFIX_ADD_TEST = "/edata/ADReSS-IS2020-data/transcription/test/"
TRANS_DICT = {PREFIX_CON: "con", PREFIX_DEM: "dem",
              PREFIX_BIRD: "bird", PREFIX_CHILD_IMPAIRED: "imparied",
              PREFIX_CHILD_NORMAL: "normal", PERFIX_ADD_TRAIN_CON: "add_train_con",
              PERFIX_ADD_TRAIN_DEM: "add_train_dem", PERFIX_ADD_TEST: "add_test"}
DEVICE = "cuda"
USE_GPU = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# check folders
if not os.path.isdir("../results/"):
    os.mkdir("../results/")
if not os.path.isdir("../results/cache-original/"):
    os.mkdir("../results/cache-original/")
if not os.path.exist("../results/evals/"):
    os.mkdir("../results/evals/")
if not os.path.exists("../results/text/"):
    os.mkdir("../results/text/")
# address dataset
if not os.path.exists("data/address_train_full.tsv") and \
    not os.path.exists("data/address_test_full.tsv") and \
        not os.path.exists("data/bird_frame.tsv"):
    train_con = read_data(PERFIX_ADD_TRAIN_CON, TRANS_DICT[PERFIX_ADD_TRAIN_CON])
    train_dem = read_data(PERFIX_ADD_TRAIN_DEM, TRANS_DICT[PERFIX_ADD_TRAIN_DEM])
    test_frame = read_data(PERFIX_ADD_TEST, TRANS_DICT[PERFIX_ADD_TEST])
    train_frame = train_con.append(train_dem)
    train_frame = train_frame.sample(frac=1)
    test_frame = test_frame.sample(frac=1)
    # save transcripts as local csv file
    train_frame.to_csv("data/address_train_full.tsv", index=False, sep="\t")
    test_frame.to_csv("data/address_test_full.tsv", index=False, sep="\t")
# bird dataset
bird_frame = read_data(PREFIX_BIRD, TRANS_DICT[PREFIX_BIRD])
bird_frame.to_csv("data/bird_frame.tsv", index=False, sep="\t")
train_frame = pd.read_csv("data/address_train_full.tsv", sep="\t")
test_frame = pd.read_csv("data/address_test_full.tsv", sep="\t")
# subset transcripts with >20 mmse
train_mild_frame = train_frame[train_frame["mmse"] > 20]
print("train mild dataset shape {}".format(train_mild_frame.shape))
test_mild_frame = test_frame[test_frame["mmse"] > 20]
print("test mild dataset shape {}".format(test_mild_frame.shape))
train_mild_frame.to_csv("data/address_train_mild.tsv", index=False, sep="\t")
test_mild_frame.to_csv("data/address_test_mild.tsv", index=False, sep="\t")
# subset transcripts with >24 mmse
train_slight_frame = train_frame[train_frame["mmse"] > 24]
print("train slight dataset shape {}".format(train_slight_frame.shape))
test_slight_frame = test_frame[test_frame["mmse"] > 24]
print("test slight dataset shape {}".format(test_slight_frame.shape))
train_mild_frame.to_csv("data/address_train_slight.tsv", index=False, sep="\t")
test_mild_frame.to_csv("data/address_test_slight.tsv", index=False, sep="\t")
# WLS dataset
process_wls_data()
# evaluation on the original GPT-2 model
con_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
df = evaluate_model(train_frame, con_model, gpt_tokenizer)
df.to_csv("../results/cache-original/address_train_full.tsv", index=False, sep="\t")
df = evaluate_model(test_frame, con_model, gpt_tokenizer)
df.to_csv("../results/cache-original/address_test_full.tsv", index=False, sep="\t")
df = evaluate_model(train_mild_frame, con_model, gpt_tokenizer)
df.to_csv("../results/cache-original/address_train_mild.tsv", index=False, sep="\t")
df = evaluate_model(test_mild_frame, con_model, gpt_tokenizer)
df.to_csv("../results/cache-original/address_test_mild.tsv", index=False, sep="\t")
df = evaluate_model(train_slight_frame, con_model, gpt_tokenizer)
df.to_csv("../results/cache-original/address_train_slight.tsv", index=False, sep="\t")
df = evaluate_model(test_slight_frame, con_model, gpt_tokenizer)
df.to_csv("../results/cache-original/address_test_slight.tsv", index=False, sep="\t")

