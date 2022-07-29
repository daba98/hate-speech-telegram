import emoji
import numpy as np
import re
from langdetect import detect
import pandas as pd 

import gensim

from nltk.corpus import stopwords
stop_words_ger = stopwords.words('german')
stop_words_en = stopwords.words('english')

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

import spacy
nlp = spacy.load("de_core_news_sm")

def prepare_single_text(text):

    text = remove_hand_selected_words(text)    
    text = rermove_repeating_chars(text)
    text = emoji_2_text(text)
    text= " ".join(tw_process.pre_process_doc(text))
    text = remove_special_chars(text)
    #text = text.lower()
    text = remove_white_spaces(text)

    return(text)
	
def prepare_single_text_without_ekphrasis(text):

    text = remove_hand_selected_words(text)    
    text = rermove_repeating_chars(text)
    text = emoji_2_text(text)
    text= remove_usernames(text)
    text = remove_URLs(text)
    text = remove_special_chars(text)
    #text = text.lower()
    text = remove_white_spaces(text)

    return(text)


def remove_usernames(string):
    return re.sub('@\S+', '<name>', string)	

def remove_URLs(string):
    return re.sub('http[s]?://\S+', '<url>', string)

def get_text_processor():
    return(TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
        
        segmenter="twitter", 
        corrector="twitter", 
        
        fix_html=True,  # fix HTML tokens

        unpack_hashtags=False,  # perform word segmentation on hashtags
        unpack_contractions=False,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        tokenizer=SocialTokenizer(lowercase=False).tokenize,
        dicts=[emoticons]
    ))
tw_process = get_text_processor()

def remove_numbers(s):
    return(re.sub(" \d+", " ", s))

def lemmatize_tokens(tokens):
    txt = " ".join(tokens)
    doc = nlp(txt)
    return([token.lemma_ for token in doc])

def keepOnlyNounsAndNE(tokens):
    txt = " ".join(tokens)
    doc = nlp(txt)
    
    keep = [str(w) for w in list(doc.ents)]
    for w in doc:
        if w.pos_ in ['NOUN']:
            keep.append(str(w.text))
    return([w for w in tokens if w in keep])
 
def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def build_N_grams(text,threshold):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(text, min_count=5, threshold=threshold) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[text], threshold=threshold)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_words_bigrams = make_bigrams(text, bigram_mod)
    return(data_words_bigrams)

def remove_stopwords(tweet):
    txt = []
    for w in tweet:
        if w in stop_words_ger:
            continue
            
        if w in stop_words_en:
            continue
        txt.append(w)

    return(txt)

def remove_hand_selected_words(text):
    return(re.sub(r"\|lbr\||\|LBR\||\|AMP\||&gt;|&amp;", " ", text))

def rermove_repeating_chars(text):
    to_remove = ".,?!"
    return(re.sub("(?P<char>[" + re.escape(to_remove) + "])(?P=char)+", r"\1", text))

from emoji import EMOJI_UNICODE, UNICODE_EMOJI
UNICODE_EMOJI = {v.encode('unicode-escape').decode().replace("\\", "").lower(): "<"+k.replace(":","").replace('-','')+">" for k, v in EMOJI_UNICODE['en'].items()}

def emoji_2_text(text):
    
    text = emoji.demojize(text, delimiters=("<",">"))
    re_matches = re.findall(r"(<U\+[0-9a-zA-Z]*>)", text)
    
     
    for emoji_unicode in re_matches:
        try:
            text = text.replace(emoji_unicode, UNICODE_EMOJI[re.sub('[<>+]', '', emoji_unicode).lower()])
        except:
            text = text.replace(emoji_unicode,"")
    
        text = text.replace(emoji_unicode,"")
        
    m = re.search('<[a-z_-]*(-)[a-z_-]*>',text)
    if m is not None:
        old_emoji = m.group(0)
        new_emoji = m.group(0).replace('-','_')
        text = text.replace(old_emoji,new_emoji)
    return(text)

def remove_special_chars(text):
    return(re.sub(r"[^A-Za-z0-9\säüßöÖÄÜ<>_:!?.,\-]+", " ", text))

def detect_language(tw):
    try:
        return(detect(tw))
    except:
        return("unk")
    
def sentence_to_token(text):
    return([w.strip() for w in text.split()])

def token_to_sentence(token):
    return(" ".join([w.strip() for w in token]))

def remove_white_spaces(text):
    return(" ".join(text.split()))
