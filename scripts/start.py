"""
starter script
"""

from util_fun import read_data, evaluate_model_with_output
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PERFIX_ADD_TRAIN_CON = "/edata/ADReSS-IS2020-data/transcription/train/cc/"
PERFIX_ADD_TRAIN_DEM = "/edata/ADReSS-IS2020-data/transcription/train/cd/"
PERFIX_ADD_TEST = "/edata/ADReSS-IS2020-data/transcription/test/"
TRANS_DICT = {PERFIX_ADD_TRAIN_CON: "add_train_con",
              PERFIX_ADD_TRAIN_DEM: "add_train_dem",
              PERFIX_ADD_TEST: "add_test"}

con_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
test_frame = read_data(PERFIX_ADD_TEST, TRANS_DICT[PERFIX_ADD_TEST])
train_con = read_data(PERFIX_ADD_TRAIN_CON, TRANS_DICT[PERFIX_ADD_TRAIN_CON])
train_dem = read_data(PERFIX_ADD_TRAIN_DEM, TRANS_DICT[PERFIX_ADD_TRAIN_DEM])
train_frame = train_con.append(train_dem)
train_frame = train_frame.sample(frac=1)
test_frame = test_frame.sample(frac=1)
# save transcripts as local csv file
train_frame.to_csv("address_train.csv", index=False)
test_frame.to_csv("address_test.csv", index=False)
evaluate_model_with_output(train_frame, con_model,
                           gpt_tokenizer, "../results/cache-original/", "con_train.json")
evaluate_model_with_output(test_frame, con_model,
                           gpt_tokenizer, "../results/cache-original/", "con_test.json")

