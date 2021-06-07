"""
Utility functions
"""

import os
import gc
import re
import glob
import warnings
import pickle
import math
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from sklearn.metrics import roc_curve, auc


warnings.filterwarnings('ignore')
DEVICE = "cuda"
USE_GPU = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def check_folder(folder_path):
    """
    check if folder exists

    :param folder_path: path to the folder
    :type folder_path: str
    """
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def str2bool(v):
    """
    convert user input into boolean value

    :param v: user input of true or false
    :type v: str
    :raises argparse.ArgumentTypeError: Boolean value expected.
    :return: a boolean value
    :rtype: bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_file(file_path):
    """
    if the file exists, then delete it

    :param file_path: the full path to the fle
    :type file_path: str
    """
    if os.path.exists(file_path):
        os.system("rm " + file_path)


def get_word_dist(input_frame, trans_col):
    """
    get the log word frequency distrubtion on the input frame, 
    :param pandas.DataFrame input_frame: the input frame with transcripts to evalute
    :param str trans_col: the column name represents transcript text
    :return: the log frequency for the transcript tokens
    :rtype: dict
    """
    word_dist = load_word_dist()
    log_lf = {}
    trans = input_frame[trans_col].values.tolist()
    for tran in trans:
        try:
            words = tran.split(" ")
            for word in set(words):
                log_lf[word.lower()] = get_word_lf(word, word_dist)
        except AttributeError:
            continue
    log_lf = sorted(log_lf.items(), key=lambda item: item[1])
    return log_lf


def get_word_lf(token, word_dist):
    """
    get the log lexical frequency for a specifc word
    :param str token: the word for calcualtion
    :param dict word_dict: the dictionary for the word raw frequency
    :return: the log lexical frequency for the word
    :rtype: float
    """
    return np.log(word_dist[token.lower()])


def load_word_dist():
    """
    load Subtlex.US.text file for estimating word frequency distribution
    save it to local file
    return the word frequency distribution
    :rtype: nltk.probability.FreqDist
    """
    if os.path.exists("word_dist.pkl"):
        with open("word_dist.pkl", "rb") as f:
            word_dist = pickle.load(f)
        return word_dist
    else:
        word_dist = FreqDist()
        sys.stdout.write("estimating frequency distribution...\n")
        with open("Subtlex.US.text", "r") as c:
            lines = c.readlines()
            for ln in lines:
                words = ln.split()
                for word in words:
                    word_dist[word.lower()] += 1
        sys.stdout.write("done\n")
        with open("word_dist.pkl", "wb") as f:
            pickle.dump(word_dist, f)
        return word_dist


def process_wls_data():
    """
    pre-process WLS meta data and transcripts
    code borrowed from Yue
    return full dataset for WLS transcript
    :return: the full dataset of WLS
    :rtype: pd.DataFrame
    """
    # Yue's code
    if not os.path.exists("data/wls_totoal.csv"):
        _dir_wls = '/edata/wls/wls_activity_transcripts/'
        wls_control_input = [open(filename, encoding='cp1252').read() for filename in glob.glob(_dir_wls+"*.txt")]
        clean = []
        for i, line in enumerate (wls_control_input):
            line = re.sub(r'\&\=clears\s+throat',r' ',line) # throat clears
            line = re.sub(r'(\w+)\((\w+)\)',r'\1\2',line) # open parentheses e.g, comin(g)
            line = re.sub(r'\((\w+)\)(\w+)',r'\1\2',line) # open parentheses e.g, (be)coming
            line = re.sub(r'\s+\w+\s+\[\:\s+([^\]]+)\]',r' \1 ', line) # open square brackets eg. [: overflowing] - error replacements
            line = re.sub(r'\&\w+\s+',r' ', line) # remove disfluencies prefixed with "&"
            line = re.sub(r'xxx',r' ', line) # remove unitelligible words
            line = re.sub(r'\(\.+\)',r' ', line) # remove pauses eg. (.) or (..)
            line = re.sub(r'\[\/+\]',r' ', line) # remove forward slashes in square brackets
            line = re.sub(r'\&\=\S+\s+',r' ', line) # remove noise indicators eg. &=breath

            line = re.sub(r'\*PAR\:',r' ', line) # remove turn identifiers
            line = re.sub(r'\[(\*|\+|\%)[^\]]+\]',r' ', line) # remove star or plus and material inside square brackets indicating an error code
            line = re.sub(r'\[(\=\?)[^\]]+\]',r' ', line)

            line = re.sub(r'[^A-Za-z\n \']','',line) # finally remove all non alpha characters

            #line = "<s> "+ line + "</s>" # format with utterance start and end symbols
            line = re.sub(r'\s+',' ',line) # replace multiple spaces with a single space
            line = line.lower() # lowercase
            clean.append(line)
        wls_control_text = clean
        wls_control_names = [filename.replace('/edata/wls/wls_activity_transcripts/','') for filename in glob.glob(_dir_wls+"/*.txt")]
        wls_control_frame = pd.DataFrame(list(zip(wls_control_names,wls_control_text)),columns=['file','text'])
        wls_meta = pd.read_csv('/edata/wls/wls_metadata/WLS_data_05202020.csv')
        var = ['idpriv', 'rtype', 'z_gi206re', 'z_gi210rec','z_gi306re', 'z_gi310rec',
                'z_gx209lre', 'z_gx209sre', 
                'z_gx361re', 'z_gx362are', 'z_gx362bre', 'z_gv032are', 
                'z_gv032bre', 'z_ga003re', 'z_gb001re', 'z_sexrsp']
        wls_meta = wls_meta[var]
        wls_meta.columns = ['idpriv', 'rtype','flu_letter', 'flu_letterScore',
                            'flu_animal', 'flu_animalScore', 'cog_level', 'cog_score',
                            'mental_diag', 'mental_icd1', 'mental_icd2', 'ill_cond1',
                            'ill_cond2', 'age', 'edu', 'sex']
        wls_meta = wls_meta.dropna(how='any').reset_index(drop=True)
        flu_animalCat = []
        for i in range(len(wls_meta)):
            if wls_meta['flu_animalScore'][i] == 'NOT ASCERTAINED':
                lettercat = None
            elif wls_meta['flu_animalScore'][i] == 'refused':
                lettercat = None
            elif int(wls_meta['flu_animalScore'][i]) < 12:
                lettercat = 1
            elif int(wls_meta['flu_animalScore'][i]) < 14:
                if wls_meta['age'][i] > 80:
                    lettercat = 0
                elif wls_meta['age'][i] == 80:
                    lettercat = 0
                else:
                    lettercat = 1
            elif int(wls_meta['flu_animalScore'][i]) < 16:
                if wls_meta['age'][i] > 60:
                    lettercat = 0
                elif wls_meta['age'][i] == 60:
                    lettercat = 0
                else:
                    lettercat = 1
            else:
                lettercat = 0
            flu_animalCat.append(lettercat)
        wls_meta['flu_animalCat'] = flu_animalCat
        wls_miid = []
        for id in wls_control_frame['file']:
            wls_miid.append(id[:7])
        wls_id = []
        for id in wls_control_frame['file']:
            wls_id.append(id[:9])
        wls_control_frame['id'] = wls_id
        wls_control_frame['miid'] = wls_miid
        wls_control_frame['label'] = np.zeros(len(wls_control_frame))
        wls_control_frame['mmse'] = np.nan
        wls_control_frame['mmse_Fritsch'] = np.nan
        wls_meta_noa = wls_meta[wls_meta['flu_animalCat']== 0]
        wls_meta_nom = wls_meta_noa.loc[wls_meta_noa['mental_diag']=='no']
        miid = []
        miid = wls_meta_nom['idpriv'].astype(str) + wls_meta_nom['rtype']
        wls_control_frame_n = wls_control_frame[wls_control_frame['miid'].isin(miid)]
        wls_meta_nom['miid'] = miid
        wls_frame_total = pd.merge(wls_control_frame_n, wls_meta_nom, on=['miid'])
        wls_frame_total.rename(columns={'id_x':'id'}, inplace=True)
        wls_frame_total.to_csv("data/wls_totoal.tsv", index=False, sep="\t")


def get_pid(prefix_path):
    """
    get participant ids for DemBank & ADR dataset
    return the pid as dataframe

    :param prefix_path: the prefix path to .cha files for DemBank files
    :type prefix_path: str
    :return: the dataframe with file and pid as columns
    :rtype: pd.DataFrame
    """
    pid_frame = pd.DataFrame(columns=["chat_id", "pid"])
    for _, _, files in os.walk(prefix_path):
        for file in files:
            # iterate .cha files only
            if file.endswith(".cha"):
                with open(os.path.join(prefix_path, file), "r") as chat_file:
                    file_id = file.split(".")[0]
                    lines = chat_file.readline()
                    while lines:
                        if re.match(r'\@PID\:', lines):
                            pid = lines.split("-")[1]
                            pid_frame = pid_frame.append({"chat_id":file_id,
                                                        "pid":pid},
                                                        ignore_index=True)
                        lines = chat_file.readline()
    return pid_frame


def str2array(input_str):
    """
    transform the read-in str from dataframe to ndarray

    :param input_str: the input str from dataframe
    :type input_str: str
    """
    tr_list = input_str[1:-1].split(" ")
    tr_list = [item for item in tr_list if item]
    tr_list = [int(item) for item in tr_list]
    return tr_list


def get_db_dataset():
    """
    get dementia bank dataset with 99 control, 169 dementia cases

    """
    total_pid = []
    fold_file = pd.read_csv("db_folds.txt")
    fold_file["label"] = np.where(fold_file["label"] == "dem", 1, 0)
    for i in range(5):
        cur_dem_train = fold_file.loc[(fold_file["fold"] == i) & (fold_file["label"] == 1)]["trainfiles"].values.tolist()[0]
        cur_con_train = fold_file.loc[(fold_file["fold"] == i) & (fold_file["label"] == 0)]["trainfiles"].values.tolist()[0]
        cur_dem_test = fold_file.loc[(fold_file["fold"] == i) & (fold_file["label"] == 1)]["testfiles"].values.tolist()[0]
        cur_con_test = fold_file.loc[(fold_file["fold"] == i) & (fold_file["label"] == 0)]["testfiles"].values.tolist()[0]
        train_fold = str2array(cur_dem_train) + str2array(cur_con_train)
        test_fold = str2array(cur_dem_test) + str2array(cur_con_test)
        total_pid.extend(train_fold)
        total_pid.extend(test_fold)
    db = pd.read_csv("data/db_full.tsv", sep="\t")
    db = db.loc[db["file"].isin(total_pid)]
    db.to_csv("data/db.tsv", sep="\t", index=False)
    

def get_dbca_dataset():
    """
    get pre-processed DBCA dataset
    return the DBCA dataset
    """
    if os.path.exists("data/dbca.tsv"):
        dbca = pd.read_csv("data/dbca.tsv", sep="\t")
    else:
        # get DementiaBank dataset
        prefix_con = "/edata/lixx3013/dementia-data/DementiaBank//DemBank/Control/99/"
        prefix_dem = "/edata/lixx3013/dementia-data/DementiaBank/DemBank/Dementia/169/"
        db_con = read_data(prefix_con, "con")
        db_dem = read_data(prefix_dem, "dem")
        db_full = db_con.append(db_dem)
        db_full = db_full.sample(frac=1)
        # get participant ids for Dementia Bank dataset
        db_con = "/edata/lixx3013/dementia-data/DementiaBank/DemBank/Control/cookie"
        con_pid = get_pid(db_con)
        db_dem = "/edata/lixx3013/dementia-data/DementiaBank/DemBank/Dementia/cookie"
        dem_pid = get_pid(db_dem)
        db_pid = con_pid.append(dem_pid)
        db_full["chat_id"] = db_full["file"].apply(lambda x: x[:5])
        db_full["id"] = db_full["file"].apply(lambda x: x[:3])
        db_full = pd.merge(db_full, db_pid, on="chat_id")
        # get participant ids for ADReSS dataset
        ad_train_cc_pid = get_pid('/edata/lixx3013/artificial-impairment/data/ADReSS/ADReSS-IS2020-data/train/transcription/cc/')
        ad_train_cd_pid = get_pid('/edata/lixx3013/artificial-impairment/data/ADReSS/ADReSS-IS2020-data/train/transcription/cd/')
        ad_test_pid = get_pid('/edata/lixx3013/artificial-impairment/data/ADReSS/ADReSS-IS2020-data/test/transcription/')
        adr_pid = ad_train_cc_pid["pid"].values.tolist() + ad_train_cd_pid["pid"].values.tolist() + ad_test_pid["pid"].values.tolist()
        # get id from ADR dataset
        adr_train = pd.read_csv("data/adress_train_full.tsv", sep="\t")
        adr_test = pd.read_csv("data/adress_test_full.tsv", sep="\t")
        adr_full = adr_train.append(adr_test)
        adr_full["id"] = adr_full["file"].apply(lambda x: x[:3])
        db_grouped = db_full.groupby(by=['id'])['pid'].apply(' '.join).reset_index()
        # remove adr ids
        dbca = pd.DataFrame()
        for ids in db_grouped['pid']:
            idlist = ids.split()
            skip = 0
            for i in idlist:
                if i in adr_pid:
                    skip = 1
            if skip == 0:
                compid = db_grouped[db_grouped['pid'] == ids]['id'].values[0] 
                dbca = dbca.append(db_full[db_full['id'] == compid])
        print("Controls: ", len(set(dbca[dbca['label'] == 0]['id'])))
        print("Dementia: ", len(set(dbca[dbca['label'] == 1]['id'])))
        # add MMSE
        db_meta_data_file = "/edata/lixx3013/hammer-nets/scripts/data/demantiabank_metadata.csv"
        db_meta_df = pd.read_csv(db_meta_data_file)
        db_meta_df['id'] = db_meta_df['id'].astype(str).str.zfill(3)
        dbca["id"] = dbca["id"].astype(str).str.zfill(3)
        dbca = pd.merge(dbca, db_meta_df, on='id')
        dbca = dbca[["text", "label", "chat_id", "id", "pid", "mms"]]
        dbca.to_csv("data/dbca.tsv", sep="\t", index=False)
    return dbca


def read_data(prefix_path, data_type):
    """
    read the data into DataFrame
    :param str prefix_path: the prefix path to the dataset
    :param str data_type: the type of the data, including:
                          - con,
                          - dem,
                          - bird,
                          - impaired,
                          - normal,
                          - add_train_con
                          - add_train_dem
                          - add_test_con
                          - add_test_dem
    :return: the dataframe for the dataset
    :rtype: pandas.DataFrame
    """
    trans_df = pd.DataFrame(columns=["file", "text", "label"])
    for filename in os.listdir(prefix_path):
        if filename.endswith(".txt"):
            if data_type in ("con", "dem"):
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        line = line.replace("<s>", "")
                        line = line.replace("</s>", "")
                        line = line.strip()
                        doc_string += line
                        doc_string += ". "
                        doc_string += line
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                    if data_type == "con":
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 0},
                                                   ignore_index=True)
                    else:
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 1},
                                                   ignore_index=True)
            elif data_type in ("normal", "impaired"):
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        line = line.strip()
                        doc_string += line
                        doc_string += ". "
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                    if data_type == "normal":
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 0},
                                                   ignore_index=True)
                    else:
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 1},
                                                   ignore_index=True)
            elif data_type == "bird":
                # bird
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        line = line.strip()
                        doc_string += line
                        doc_string += " "
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                    if filename == "mct_1_0.txt":
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 0},
                                                    ignore_index=True)
                    else:
                        trans_df = trans_df.append({"file": filename,
                                                    "text": doc_string,
                                                    "label": 1},
                                                    ignore_index=True)
            else:
                meta = pd.read_csv('/edata/ADReSS-IS2020-data/meta_data_test.txt', sep=';')
                # address
                with open(os.path.join(prefix_path, filename),
                          errors="ignore", encoding="utf-8") as f:
                    doc_string = ""
                    for line in f:
                        line = line.strip()
                        doc_string += line
                        doc_string += ". "
                    doc_string = doc_string.replace("\n", "")
                    doc_string = doc_string.strip()
                if data_type == "add_train_con":
                    cc_meta = pd.read_csv("/edata/ADReSS-IS2020-data/train/cc_meta_data.txt", sep=";")
                    mmse = cc_meta[cc_meta['ID   '] == filename[:-4]+' ']['mmse'].values[0]
                    # str to int, na to np.nan
                    try:
                        mmse = int(mmse)
                    except ValueError:
                        if mmse == " NA":
                            mmse = np.nan
                    trans_df = trans_df.append({"file": filename,
                                                "text": doc_string,
                                                "label": 0,
                                                "mmse": mmse},
                                                ignore_index=True)
                elif data_type == "add_train_dem":
                    cd_meta = pd.read_csv("/edata/ADReSS-IS2020-data/train/cd_meta_data.txt", sep=";")
                    mmse = cd_meta[cd_meta['ID   '] == filename[:-4]+' ']['mmse'].values[0]
                    mmse = cd_meta[cd_meta['ID   '] == filename[:-4]+' ']['mmse'].values[0]
                    # str to int, na to np.nan
                    try:
                        mmse = int(mmse)
                    except ValueError:
                        if mmse == " NA":
                            mmse = np.nan
                    trans_df = trans_df.append({"file": filename,
                                                "text": doc_string,
                                                "label": 1,
                                                "mmse": mmse},
                                                ignore_index=True)
                else:
                    label = meta[meta['ID   '] == filename[:-4]+' ']['Label '].values[0]
                    mmse = meta[meta['ID   '] == filename[:-4]+' ']['mmse'].values[0]
                    trans_df = trans_df.append({"file": filename,
                                                "text": doc_string,
                                                "label": label,
                                                "mmse": mmse},
                                                ignore_index=True)
    return trans_df


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
    text = re.sub(r"\_", "", text)
    # remove unwanted whitespaces between words
    text = re.sub(r'\s+', " ", text)
    text = text.strip()
    return text


def process_ccc():
    """
    read and pre-process ccc dataset:
        - read the ccc dataset pickle file
        - clean the transcript
        - save it to a local file
    :return: ccc cleaned dataset
    :rtype: pd.DataFrame
    """
    with open("/edata/dementia_interviewee_only_0527.pkl", "rb") as f:
        df = pickle.load(f)
    df["label"] = np.where(df["dementia"], 1, 0)
    df = df[["ParticipantID", "Transcript", "label"]]
    # rename columns to keep same track with ADReSS
    df.columns = ["file", "text", "label"]
    df["text"] = df["text"].apply(clean_ccc_text)
    # drop empty rows if any
    df = df[df["text"].str.len() > 0]
    df.to_csv("data/ccc_cleaned.tsv", sep="\t", index=False)


def model_driver(input_text, model, tokenizer):
    """
    get the output from the model

    :param input_text: the transcript from a row of dataset
    :type input_text: Pandas.Series
    :param model: the GPT-2 model for evaluation
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    """
    encodings = tokenizer(input_text, return_tensors="pt",
                          return_attention_mask=True,
                          add_special_tokens=True,
                          truncation=True,
                          max_length=1024)
    if USE_GPU:
        input_ids = encodings["input_ids"]
        att_mask = encodings["attention_mask"]
        input_ids = input_ids.to(DEVICE)
        att_mask = att_mask.to(DEVICE)
        outputs = model(input_ids, attention_mask=att_mask, labels=input_ids)
        del input_ids, att_mask
        gc.collect()
        return outputs
    else:
        input_ids = encodings["input_ids"]
        att_mask = encodings["attention_mask"]
        outputs = model(input_ids, attention_mask=att_mask, labels=input_ids)
        del input_ids, att_mask
        gc.collect()
        return outputs


def evaluate_model(test_frame, model, tokenizer):
    """
    evaluate the test dataframe with modified model,
    if output is True, then save the results to local file,
    else returns the results as a dataframe

    :param test_frame: the input dataset for evaluation
    :type test_frame: pandas.DataFrame
    :param model: the modified model
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the gpt-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :return: a dataframe
    :rtype: pandas.DataFrame
    """
    model.eval()
    res_df = pd.DataFrame(columns=["file", "label", "perplexity", "mmse"])
    columns = test_frame.columns
    if USE_GPU:
        model.to(DEVICE)
    for _, row in test_frame.iterrows():
        with torch.no_grad():
            trans = row["text"]
            outputs = model_driver(trans, model, tokenizer)
            # calculate perplexity score
            perp = math.exp(outputs[0].item())
            # add MMSE column
            if "mmse" in columns:
                eval_dict = {"file": row["file"],
                             "label": row["label"],
                             "perplexity": perp,
                             "mmse": row["mmse"]}
            else:
                eval_dict = {"file": row["file"],
                             "label": row["label"],
                             "perplexity": perp,
                             "mmse": 0}
            res_df = res_df.append(eval_dict, ignore_index=True)
            del outputs
            gc.collect()
    return res_df


def calculate_accuracy(labels, perp):
    """
    calculate accuracy given labels and perpelxity scores

    :param labels: the transcript labels
    :type labels: list
    :param perp: the perplexity scores
    :type perp: list
    :return: accuracy and acu
    """
    fpr, tpr, _ = roc_curve(labels, perp)
    fnr = 1 - tpr
    tnr = 1 - fpr
    auc_level = auc(fpr, tpr)
    prevalence = np.count_nonzero(labels)/len(labels)
    eer_point = np.nanargmin(np.absolute((fnr - fpr)))
    tpr_at_eer = tpr[eer_point]
    tnr_at_eer = tnr[eer_point]
    accuracy = tpr_at_eer * prevalence + tnr_at_eer * (1-prevalence)
    return accuracy, auc_level


def calculate_metrics(res_dict, model_dem, tokenizer,
                      input_frame, con_res_df):
    """
    calculate AUC and accuracy for control, dementia, c/d model
    return the evaluation result

    :param res_dict: a dictionary to store all metrics
    :type res_dict: dict
    :param model_dem: the modified model serving as dementia
    :type model_dem: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the gpt-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :param input_frame: the dataframe to be evaluated
    :type input_frame: pd.DataFrame
    :param con_res_df: the groupby control model evaluation result dataframe
    :type con_res_df: pd.DataFrame
    """
    dem_res_df = evaluate_model(input_frame, model_dem, tokenizer)
    dem_res_df = dem_res_df.groupby(["file", "label", "mmse"])["perplexity"].mean().reset_index()
    dem_res_df.rename(columns={"perplexity": "dem_ppl"}, inplace=True)
    full_res = pd.merge(con_res_df, dem_res_df, on=["file", "label", "mmse"])
    mmse = full_res["mmse"].values.tolist()
    # control model AUC, accuracy, cor
    labels = full_res["label"].values.tolist()
    con_ppl = full_res["con_ppl"].values.tolist()
    con_accu, con_auc = calculate_accuracy(labels, con_ppl)
    c_r = np.corrcoef(con_ppl, mmse)[1,0]
    res_dict["con_auc"].append(round(con_auc, 3))
    res_dict["con_accu"].append(round(con_accu, 3))
    res_dict["con_cor"].append(round(c_r, 3))
    res_dict["con_ppl"].append((np.mean(con_ppl)))
    # dementia model AUC, accuracy, cor
    dem_ppl = full_res["dem_ppl"].values.tolist()
    dem_accu, dem_auc = calculate_accuracy(labels, dem_ppl)
    d_r = np.corrcoef(dem_ppl, mmse)[1,0]
    res_dict["dem_auc"].append(round(dem_auc, 3))
    res_dict["dem_accu"].append(round(dem_accu, 3))
    res_dict["dem_cor"].append(round(d_r, 3))
    res_dict["dem_ppl"].append(np.mean(dem_ppl))
    # c/d model AUC, accuracy, cor
    ratio_ppl = full_res["con_ppl"]/full_res["dem_ppl"]
    ratio_ppl = ratio_ppl.values.tolist()
    ratio_accu, ratio_auc = calculate_accuracy(labels, ratio_ppl)
    ratio_r = np.corrcoef(ratio_ppl, mmse)[1,0]
    res_dict["ratio_auc"].append(round(ratio_auc, 3))
    res_dict["ratio_accu"].append(round(ratio_accu, 3))
    res_dict["ratio_cor"].append(round(ratio_r, 3))
    res_dict["ratio_ppl"].append((np.mean(ratio_ppl)))
    # log(c)-log(d) AUC, accuracy, cor
    '''
    log_ppl = np.log(full_res["con_ppl"])-np.log(full_res["dem_ppl"])
    log_ppl = log_ppl.values.tolist()
    log_accu, log_auc = calculate_accuracy(labels, log_ppl)
    log_r = np.corrcoef(log_ppl, mmse)[1,0]
    res_dict["log_auc"].append(round(log_auc, 3))
    res_dict["log_accu"].append(round(log_accu, 3))
    res_dict["log_cor"].append(round(log_r, 3))
    res_dict["log_ppl"].append((np.mean(log_ppl)))
    '''
    # norm ppl diff
    norm_ppl = (np.log(full_res["con_ppl"]) - np.log(full_res["dem_ppl"]))/np.log(full_res["con_ppl"])
    norm_ppl = norm_ppl.values.tolist()
    norm_accu, norm_auc = calculate_accuracy(labels, norm_ppl)
    norm_r = np.corrcoef(norm_ppl, mmse)[1,0]
    res_dict["norm_auc"].append(round(norm_auc, 3))
    res_dict["norm_accu"].append(round(norm_accu, 3))
    res_dict["norm_cor"].append(round(norm_r, 3))
    res_dict["norm_ppl"].append(np.mean(norm_ppl))
    return res_dict


def break_attn_heads_by_layer(zero_type, model, share, layer):
    """
    set certain percentage attention heads to zero at specific layer
    return the modified model
    :param zero_type: the type for zeroing attention heads,
                      'random', 'first' and 'shuffle' are supported
    :type zero_type: str
    :param model: the oringal GPT-2 model to be modified
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param share: % of attention heads to be zeroed,
                  25%, 50%, and 100% are supported
    :type share: int
    :param layer: the specific layer to be modified,
                  ranging from 0 to 11
    :type layer: int

    :return: the modified model
    :rtype: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    # zeroing both weights and bias
    head_offsets = [1536, 1536+64, 1536+128, 1536+192, 1536+256,
                    1536+320, 1536+384, 1536+448, 1536+512,
                    1536+576, 1536+640, 1536+704]
    batch = 64
    with torch.no_grad():
        if zero_type == 'random':
            np.random.seed(42)
            torch.manual_seed(42)
            # Serguei's approach to reduce running time
            for head in head_offsets:
                # update to unique random integers
                rnd_index = np.random.choice(range(head, head+64), int(batch*(share/100)), replace=False)
                for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                    model.transformer.h[layer].attn.c_attn.weight[row][rnd_index] = \
                        model.transformer.h[layer].attn.c_attn.weight[row][rnd_index].mul(0.0)
                    model.transformer.h[layer].attn.c_attn.bias[rnd_index] = \
                        model.transformer.h[layer].attn.c_attn.bias[rnd_index].mul(0.0)
            return model
        elif zero_type == 'first':
            offset = int(batch*(share/100))
            for head in head_offsets:
                for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                    model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset] = \
                        model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset].mul(0.0)
                    model.transformer.h[layer].attn.c_attn.bias[head:head+offset] = \
                        model.transformer.h[layer].attn.c_attn.bias[head:head+offset].mul(0.0)
            return model
        elif zero_type == 'shuffle':
            offset = int(64*(share/100))
            for head in head_offsets:
                for row in range(0,model.transformer.h[layer].attn.c_attn.weight.size()[0]):
                    np.random.shuffle(model.transformer.h[layer].attn.c_attn.weight[row][head:head+offset])
                    np.random.shuffle(model.transformer.h[layer].attn.c_attn.bias[row][head:head+offset])
            return model
        else:
            raise ValueError("zeroing type is not supported!")


def accumu_model_driver(model, share, zero_style, num_layers):
    """
    the driver function for breaking GPT-2 model
    :param model: the oringal GPT-2 model to be modified
    :type model: transformers.modeling_gpt2.GPT2LMHeadModel
    :param share: % of attention heads to be zeroed
    :type share: int
    :param zero_style: the style of zeroing attention heads
    :type zero_style: str
    :param num_layers: numer of layers to be zeroed
    :type num_layers: int
    :return: the modified model
    :rtype: transformers.modeling_gpt2.GPT2LMHeadModel
    """
    if num_layers > 13:
        raise ValueError("GPT-2 model only has 12 layers")
    for i in range(0, num_layers):
        model = break_attn_heads_by_layer(zero_style, model, share, i)
    return model


def generate_texts(model_con, model_dem, tokenizer, bird_sents):
    """
    generate additional 20 tokens for each sentence of healthy bird transcript,
    find the highest non-empty beam result for both dementia and control model
    return the output as a dataframe
    :param model_con: the control model
    :type model_con: transformers.modeling_gpt2.GPT2LMHeadModel
    :param model_dem: the dementia model
    :type model_dem: transformers.modeling_gpt2.GPT2LMHeadModel
    :param tokenizer: the GPT-2 tokenizer
    :type tokenizer: transformers.tokenization_gpt2.GPT2Tokenizer
    :param bird_sents: the sentences from bird control transcript
    :type bird_df: list
    """
    torch.manual_seed(42)
    np.random.seed(42)
    # in case of duplicate outputs
    out_df = pd.DataFrame(columns=["sentence", "control", "dementia"])
    '''
    bird_df = pd.read_csv("data/bird_frame.tsv", sep="\t")
    bird_all = bird_df[bird_df["file"] == "mct_all.txt"]["text"].values.tolist()[0]
    bird_sents = sent_tokenize(bird_all)
    '''
    # iterate all senteces from healthy bird transcript
    for sent in bird_sents:
        encoded_input = tokenizer.encode(sent, add_special_tokens=True, return_tensors="pt")
        prompt_length = len(tokenizer.decode(encoded_input[0], skip_special_tokens=True,
                            clean_up_tokenization_spaces=True))
        con_output = model_con.generate(encoded_input, top_p=0.9, temperature=1,
                                        max_time=5, num_beams=5, no_repeat_ngram_size=1,
                                        num_return_sequences=5, early_stopping=False,
                                        max_length=prompt_length+20)
        dem_output = model_dem.generate(encoded_input, top_p=0.9, temperature=1,
                                        max_time=5, num_beams=5, no_repeat_ngram_size=1,
                                        num_return_sequences=5, early_stopping=False,
                                        max_length=prompt_length+20)
        # find the non-empty output for both model
        for con_beam, dem_beam in zip(con_output, dem_output):
            # control model output
            con_beam = tokenizer.decode(con_beam, skip_special_tokens=True)[prompt_length:]
            con_beam = re.sub(r"\s+", " ", con_beam, flags=re.UNICODE)
            con_beam = re.sub('"', "", con_beam)
            # dementia model output
            dem_beam = tokenizer.decode(dem_beam, skip_special_tokens=True)[prompt_length:]
            dem_beam = re.sub(r"\s+", " ", dem_beam, flags=re.UNICODE)
            dem_beam = re.sub('"', "", dem_beam)
            # check if outputs are empty
            # both of them are not empty
            if con_beam.strip() and dem_beam.strip():
                out_dict = {"sentence": sent, "control": con_beam, "dementia": dem_beam}
                out_df = out_df.append(out_dict, ignore_index=True)
                break
    del model_con, model_dem
    gc.collect()
    return out_df