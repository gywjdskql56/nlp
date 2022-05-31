from transformers import BertModel
# from tokenization_kobert import KoBertTokenizer
from transformers import pipeline
import os


# label 0 : negative
# label 1 : positive
os.environ["CURL_CA_BUNDLE"]=""
kobert_classifier = pipeline("sentiment-analysis", model='monologg/kobert')
jybert_classifier = pipeline("sentiment-analysis", model="jaehyeong/koelectra-base-v3-generalized-sentiment-analysis")
def cal_kobert_score(sentence):
    # print(sentence)
    score = kobert_classifier(sentence)
    neg_pos = -1 if score[0]['label']=='LABEL_0' else 1
    print(sentence, '===', score[0]['score'] * neg_pos)
    return score[0]['score'] * neg_pos

def cal_jybert_score(sentence): # https://huggingface.co/jaehyeong/koelectra-base-v3-generalized-sentiment-analysis
    # print(sentence)
    score = jybert_classifier(sentence)
    neg_pos = -1 if score[0]['label']=='0' else 1
    print(sentence, '===', score[0]['score'] * neg_pos)
    return score[0]['score'] * neg_pos
# sentence = '안녕하세요, 반갑습니다.'
# kobert_classifier = pipeline("sentiment-analysis", model='monologg/kobert')
# score = kobert_classifier(sentence)
# tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
# print(1)