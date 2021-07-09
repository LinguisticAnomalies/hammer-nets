'''
script for calculating perpelxity of GPT-2 and GPT-D on 
cookie theft and childes transcripts
'''

import sys
import os
import re
from datetime import datetime
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util_fun import accumu_model_driver, evaluate_model


def extract_from_chat(input_path, par_indicator, out_file):
    """
    iterate the directory and pre-process the .chat files,
    save the transcript into dataframe,
    save the dataframe to local file

    :param input_path: the path to the transcript folder
    :type input_path: str
    :param par_indicator: the string pattern of participants, including 'CHI' or 'PAR'
    :type par_indicator: str
    :param out_file: the path to the local file
    """
    trans = pd.DataFrame(columns=["file", "label", "text"])
    for subdir, dirs, files in os.walk(input_path):
        age_group = subdir.split("/")[-1]
        for file in files:
            with open(os.path.join(subdir, file)) as fp:
                tran = []
                line = fp.readline()
                while line:
                    if re.match(r"\*"+par_indicator+":", line):
                        line = re.sub(r"\*"+par_indicator+":", "", line)
                        line = re.sub(r'\&\=clears\s+throat', r' ', line)  # throat clears
                        line = re.sub(r'(\w+)\((\w+)\)', r'\1\2', line)  # open parentheses e.g, comin(g)
                        line = re.sub(r'\((\w+)\)(\w+)', r'\1\2', line)  # open parentheses e.g, (be)coming
                        line = re.sub(r'\s+\w+\s+\[\:\s+([^\]]+)\]', r' \1 ',
                                      line)  # open square brackets eg. [: overflowing] - error replacements
                        line = re.sub(r'\&\w+\s+', r' ', line)  # remove disfluencies prefixed with "&"
                        line = re.sub(r'xxx', r' ', line)  # remove unitelligible words
                        line = re.sub(r'\(\.+\)', r' ', line)  # remove pauses eg. (.) or (..)
                        line = re.sub(r'\[\/+\]', r' ', line)  # remove forward slashes in square brackets
                        line = re.sub(r'\&\=\S+\s+', r' ', line)  # remove noise indicators eg. &=breath

                        line = re.sub(r'\*PAR\:', r' ', line)  # remove turn identifiers
                        line = re.sub(r'\[(\*|\+|\%)[^\]]+\]', r' ',
                                      line)  # remove star or plus and material inside square brackets indicating an error code
                        line = re.sub(r'\[(\=\?)[^\]]+\]', r' ', line)

                        line = re.sub(r'[^A-Za-z\n \']', '', line)  # finally remove all non alpha characters

                        # line = "<s> "+ line + "</s>" # format with utterance start and end symbols
                        line = re.sub(r'\s+', ' ', line)  # replace multiple spaces with a single space
                        line = line.lower()  # lowercase
                        #if line != '<s> </s>' and line != '<s></s>':
                        line = line.strip()
                        if line:
                            tran.append(line)
                    line = fp.readline()
                full_text = ". ".join(tran)
                full_text += "."
                record = {"file": file.split(".")[0], "label": age_group,"text": full_text}
                trans = trans.append(record, ignore_index=True)
    trans.to_csv(out_file, sep="\t", index=False)


def get_perp():
    """
    calculate perplexity from ccc and childres dataset
    """
    model_con = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dem = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model_dem = accumu_model_driver(model_dem, share=100, zero_style="first", num_layers=9)
    # load dataset
    child_normal = "data/Gillam/TD"
    extract_from_chat(child_normal, "CHI", "data/gillam_normal.tsv")
    childres = pd.read_csv("data/gillam_normal.tsv", sep="\t")
    ccc = pd.read_csv("data/ccc_cleaned.tsv", sep="\t")
    db_recall_file = "/edata/cookieReduxPostNAACL/Dementia/recall"
    extract_from_chat(db_recall_file, "PAR", "data/db_recall.tsv")
    db_recall = pd.read_csv("data/db_recall.tsv", sep="\t")
    # get perplexity
    childres_con_perp = evaluate_model(childres, model_con, gpt_tokenizer)
    childres_dem_perp = evaluate_model(childres, model_dem, gpt_tokenizer)
    recall_con_perp = evaluate_model(db_recall, model_con, gpt_tokenizer)
    recall_dem_perp = evaluate_model(db_recall, model_dem, gpt_tokenizer)
    ccc_con_perp = evaluate_model(ccc, model_con, gpt_tokenizer)
    ccc_dem_perp = evaluate_model(ccc, model_dem, gpt_tokenizer)
    # ccc dataset needs additional merge
    ccc_con_perp = ccc_con_perp.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
    ccc_dem_perp = ccc_dem_perp.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
    # rename columns
    ccc_con_perp.rename(columns={"perplexity": "con_ppl"}, inplace=True)
    ccc_dem_perp.rename(columns={"perplexity": "dem_ppl"}, inplace=True)
    childres_con_perp.rename(columns={"perplexity": "con_ppl"}, inplace=True)
    childres_dem_perp.rename(columns={"perplexity": "dem_ppl"}, inplace=True)
    recall_con_perp.rename(columns={"perplexity": "con_ppl"}, inplace=True)
    recall_dem_perp.rename(columns={"perplexity": "dem_ppl"}, inplace=True)
    # merge into full dataframe
    ccc_full = pd.merge(ccc_con_perp, ccc_dem_perp, on=["file", "label", "mmse"])
    childres_full = pd.merge(childres_con_perp, childres_dem_perp, on=["file", "label", "mmse"])
    recall_full = pd.merge(recall_con_perp, recall_dem_perp, on=["file", "label", "mmse"])
    ccc_full.to_csv("../results/ppl/ccc_perp.csv", index=False)
    childres_full.to_csv("../results/ppl/childres_perp.csv", index=False)
    recall_full.to_csv("../results/ppl/recall_perp.csv", index=False)


if __name__ == '__main__':
    start_time = datetime.now()
    get_perp()
    sys.stdout.write("finished!\n")
    sys.stdout.write("total running time: {}\n".format(datetime.now()-start_time))