"""
starter script
"""

import os
from util_fun import read_data, evaluate_model_with_output
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
# check folders
if not os.path.isdir("../results/"):
    os.mkdir("../results/")
if not os.path.isdir("../results/cache-original/"):
    os.mkdir("../results/cache-original/")
# address dataset
train_con = read_data(PERFIX_ADD_TRAIN_CON, TRANS_DICT[PERFIX_ADD_TRAIN_CON])
train_dem = read_data(PERFIX_ADD_TRAIN_DEM, TRANS_DICT[PERFIX_ADD_TRAIN_DEM])
test_frame = read_data(PERFIX_ADD_TEST, TRANS_DICT[PERFIX_ADD_TEST])
train_frame = train_con.append(train_dem)
train_frame = train_frame.sample(frac=1)
test_frame = test_frame.sample(frac=1)
# save transcripts as local csv file
train_frame.to_csv("data/address_train.csv", index=False)
test_frame.to_csv("data/address_test.csv", index=False)
# subset transcripts with >20 mmse
train_mild_frame = train_frame[train_frame["mmse"] > 20]
test_mild_frame = test_frame[test_frame["mmse"] > 20]
train_mild_frame.to_csv("data/address_train_mild.csv", index=False)
test_mild_frame.to_csv("data/address_test_mild.csv", index=False)
# WLS dataset
process_wls_data()
# bird dataset
bird_frame = read_data(PREFIX_BIRD, TRANS_DICT[PREFIX_BIRD])
bird_frame.to_csv("data/bird_frame.csv", index=False)
# evaluation on the original GPT-2 model
con_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
evaluate_model_with_output(train_frame, con_model,
                           gpt_tokenizer, "../results/cache-original/", "con_full_train.json")
evaluate_model_with_output(test_frame, con_model,
                           gpt_tokenizer, "../results/cache-original/", "con_full_test.json")
evaluate_model_with_output(train_mild_frame, con_model,
                           gpt_tokenizer, "../results/cache-original/", "con_mild_train.json")
evaluate_model_with_output(test_mild_frame, con_model,
                           gpt_tokenizer, "../results/cache-original/", "con_mild_test.json")

