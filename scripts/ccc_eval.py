'''
script for evaluating CCC dataset on merged participant id
'''


import pickle
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import evaluate_model
# use GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def clean_ccc_text(text):
    """
    basic pre-processing for CCC transcripts
    :param text: the CCC transcript for pre-processing
    :type text: str
    """
    text = text.lower().strip()
    # remove () and [] and following punctuation
    text = re.sub(r"[\(\[].*?[\)\]][?!,.]?", "", text)
    # remove ^^_____ and following punctuation
    text = re.sub(r'\^+_+\s?[?.!,]?', "", text)
    text = re.sub(r'~','-',text)
    text = re.sub(r'-',' ',text)
    text = re.sub(r'[^\x00-\x7F]+','\'',text)
    text = re.sub(r'\<|\>',' ',text)
    # remove unwanted whitespaces between words
    text = re.sub(r'\s+', " ", text)
    text = text.strip()
    return text


def process_ccc():
    """
    read and pre-process ccc dataset:
        - read the ccc dataset pickle file
        - clean the transcript
        - split into train/test set
        - compute perplexity on transcript level with vanilla GPT-2 model
        - compute average perplexity on participant id
        - merge the results into a single dataframe
        - save it to local file
    """
    with open("/edata/dementia_cleaned_withId.pkl", "rb") as f:
        df = pickle.load(f)
    df["label"] = np.where(df["dementia"], 1, 0)
    df = df[["ParticipantID", "Transcript", "label"]]
    # rename columns to keep same track with ADReSS
    df.columns = ["file", "text", "label"]
    df["text"] = df["text"].apply(clean_ccc_text)
    train_df, test_df = train_test_split(df, random_state=42, test_size=0.3)
    train_df.to_csv("data/ccc_train.tsv", sep="\t", index=False)
    test_df.to_csv("data/ccc_test.tsv", sep="\t", index=False)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    # vanilla gpt-2 model evaluation
    print("evaluating ccc training set...")
    train_res = evaluate_model(train_df, model_con, gpt_tokenizer)
    train_res = train_res.groupby(["file", "label"])["perplexity"].mean().reset_index()
    train_res.to_csv("../results/cache-original/ccc_train.tsv", index=False, sep="\t")
    print("evaluating ccc test set...")
    test_res = evaluate_model(test_df, model_con, gpt_tokenizer)
    test_res = test_res.groupby(["file", "label"])["perplexity"].mean().reset_index()
    test_res.to_csv("../results/cache-original/ccc_test.tsv", index=False, sep="\t")


if __name__ == "__main__":
    process_ccc()
