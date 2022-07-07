import re
from textwrap import wrap
import datasets
import random
import unicodedata
from tqdm import tqdm




def normalize_sents(sents, wrapper=None):
    
    html_tags = re.compile('<.*?>')
    remove_years = re.compile('\(.*?\d\)')
    remove_date = re.compile('(\d{4})?-?/?\.?\d{1,2}-?/?\.?\d{1,2}')
    remove_time = re.compile('\d{1,2}:\d{1,2}')
    remove_uri = re.compile("(http(s)?://)\S+\.\S+")
    remove_email = re.compile("\S+@\S+\.\S+")
    remove_hashtags_and_username = re.compile("(#|@)\S+")
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"
                            u"\U0001F300-\U0001F5FF"
                            u"\U0001F680-\U0001F6FF"
                            u"\U0001F1E0-\U0001F1FF"
                            "]+", flags=re.UNICODE)
    new_sents = list()
    for sent in tqdm(sents, desc="normalize data"):
        sent = str(sent)
        sent = html_tags.sub("", sent)
        sent = remove_years.sub("", sent)
        sent = remove_date.sub("", sent)
        sent = remove_time.sub("", sent)
        sent = remove_email.sub("", sent)
        sent = remove_hashtags_and_username.sub("", sent)
        sent = remove_uri.sub("", sent)
        sent = emoji_pattern.sub("", sent)
        sent = sent.translate(None, "\t\n")

        sent = sent.lstrip("1234567890.()-") # remove numbers at begin

        sent = sent.replace(" ", "\u2423")
        sent = " ".join(unicodedata.normalize('NFKD', sent))
        if wrapper is None:
            sent = wrapper(sent)
        new_sents.append(sent)
       
    return new_sents