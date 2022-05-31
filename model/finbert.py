from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import numpy as np
import pandas as pd
os.environ["CURL_CA_BUNDLE"]=""
def cal_finbert_score(sentence):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    positive = predictions[:, 0].tolist()[0]
    negative = predictions[:, 1].tolist()[0]
    neutral = predictions[:, 2].tolist()[0]
    if np.argmax([positive, negative, neutral]) == 0:
        return [{'label':'POSITIVE', 'score': positive}]
        # df.loc[idx, 'finbert_score'] = positive
    elif np.argmax([positive, negative, neutral]) == 1:
        return [{'label': 'NEGATIVE', 'score': negative}]
        # df.loc[idx, 'finbert_score'] = negative * (-1)
    else:
        return [{'label': 'NEUTRAL', 'score': neutral}]
        # df.loc[idx, 'finbert_score'] = 0
print(1)
# def add_finbert_score(file_dir):
#     tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#     model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
#     for file_name in os.listdir(file_dir):
#         print(file_name)
#         if '.xlsx' not in file_name:
#             pass
#         df = pd.read_excel(file_dir + file_name, index_col=0)
#         for idx, sentence in zip(df.index, df['sentence'].tolist()):
#             inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
#             outputs = model(**inputs)
#             predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#
#             positive = predictions[:, 0].tolist()[0]
#             negative = predictions[:, 1].tolist()[0]
#             neutral = predictions[:, 2].tolist()[0]
#             print(positive)
#             df.loc[idx, 'finbert_positive'] = positive
#             df.loc[idx, 'finbert_negative'] = negative
#             df.loc[idx, 'finbert_neutral'] = neutral
#             if np.argmax([positive, negative, neutral]) == 0:
#                 df.loc[idx, 'finbert_score'] = positive
#             elif np.argmax([positive, negative, neutral]) == 1:
#                 df.loc[idx, 'finbert_score'] = negative * (-1)
#             else:
#                 df.loc[idx, 'finbert_score'] = 0
#             # df.loc[idx, 'finbert_score'] = 0
#             df.to_excel(file_dir + file_name)
#     return df