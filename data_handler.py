"""
Module with cleaning and preprocessing functions.
"""

import pandas as pd
import numpy as np
import gensim.downloader
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import nltk
import gdown

import gc
from tqdm import tqdm


nltk.download('punkt_tab')
nltk.download('stopwords')

stop = set(stopwords.words('english'))
special_char_rmv = str.maketrans(dict.fromkeys(string.punctuation))

MAX_TOKENS = 14  # 90% cleaned_data under 14 tokens

def prepare_base_cleaned_dataset():
    """

    :return:
    """
    try:
        df = pd.read_csv("dataset(clean).csv")  # original dataset from kaggle
    except FileNotFoundError:
        # download backup from google drive
        url = "https://drive.google.com/file/d/1x-Ku2Zwhm3Z4iSeyJn69XzlUgrJ8CkzB/view?usp=sharing"
        gdown.download(url=url, output="dataset(clean).csv", fuzzy=True)
        df = pd.read_csv("dataset(clean).csv")

    cleaned_text = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        txt = row[1]

        txt = txt.translate(special_char_rmv)  # handles punctuations
        txt = txt.split()
        txt = [w for w in txt if not w in stop]  # remove stop words

        txt = ' '.join(txt)
        cleaned_text.append(txt)

    df['cleaned_text'] = cleaned_text
    df = df.loc[~df['cleaned_text'].isin(['', np.nan, np.NAN, None])] # few texts are only made of stop words, so the cleaned text is empty
    df.to_parquet("basic_cleaned.parquet", index=False)


def prepare_glove_embedding_dataset():
    """

    :return:
    """

    try:
        df = pd.read_parquet("basic_cleaned.parquet")
    except FileNotFoundError:
        # download backup from google drive
        url = "https://drive.google.com/file/d/1dCDlSNPUvnD0Z69ajIiusePUAO5_PROo/view?usp=sharing"
        gdown.download(url=url, output="basic_cleaned.parquet", fuzzy=True)
        df = pd.read_parquet("basic_cleaned.parquet")

    gloveModel = gensim.downloader.load("glove-twitter-25")  # 100MB model
    vec_size = 25

    def text_embedder(text_to_embed):
        embedded_vector = []
        cleaned_text = []  # will return function input in most cases. When word is not in glove, we use 0 vector
        for word in word_tokenize(text_to_embed.lower()):
            try:
                embedded_vector.append(np.round(gloveModel[word],5).tolist())
            except KeyError:
                embedded_vector.append(np.zeros((vec_size,)).tolist())
            cleaned_text.append(word)

        if len(embedded_vector) > MAX_TOKENS:
            embedded_vector = embedded_vector[:MAX_TOKENS]
            cleaned_text = cleaned_text[:MAX_TOKENS]
        elif len(embedded_vector) < MAX_TOKENS:
            embedded_vector = embedded_vector + [np.zeros(shape=(vec_size,)).tolist() for _ in
                                                 range(MAX_TOKENS - len(embedded_vector))]

        return embedded_vector, " ".join(cleaned_text)

    output_records = []
    cnt = 0
    file_ind = 0
    records_per_file = 1000000000  # set more than count of records, to get a single file
    for row in tqdm(df.itertuples(), total=len(df)):
        e, c = text_embedder(row[-1])
        output_records.append({
            "Emotion": row[1],
            "input_text": c,
            "embedded_vector": e
        })
        cnt += 1

        if cnt > records_per_file:
            cnt = 0
            output_df = pd.DataFrame(output_records)
            output_records = []

            output_df.to_parquet(
                f"glove-{vec_size}-embedded-{MAX_TOKENS}token_{file_ind}.parquet",
                index=False
            )
            del output_df
            gc.collect()

            file_ind += 1

    # Storing the remainder data
    if output_records:
        output_df = pd.DataFrame(output_records)
        output_df.to_parquet(
            f"glove-{vec_size}-embedded-{MAX_TOKENS}token_{file_ind}.parquet",
            index=False
        )


if __name__ == "__main__":
    prepare_base_cleaned_dataset()
    prepare_glove_embedding_dataset()