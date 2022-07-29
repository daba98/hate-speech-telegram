import hashlib 
import pymongo
import torch
import math

from pymongo_ssh import MongoSession


# connect to database
def connect_to_database(key, host='138.246.233.159'):
    session = MongoSession(
    host=host,
    port=22,
    user='ubuntu',
    key=key,
    uri='mongodb://127.0.0.1:27017')
    db = session.connection['telegram']
    
    db_channels = db.channels
    db_messages = db.messages
    db_errors = db.errors
    db_users = db.users
    db_entities = db.entities
    
    return db_channels, db_messages, db_errors, db_users, db_entities

def hashContent(field1,field2,field3):
    to_hash = field1 if field1 is not None else ''
    if field2 is not None:
        to_hash = to_hash + field2
    if field3 is not None:
        to_hash = to_hash + field3
    result = hashlib.md5(to_hash.encode()) 
    return result.hexdigest()

def predict(text,tokenizer,model):
    inputs =  tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to('cuda')
    labels = torch.tensor([1]).unsqueeze(0).cuda()
    outputs = model(**inputs, labels=labels)
    m = torch.nn.Softmax(dim=1).cuda()
    # softmax the logits
    softmaxed = m(outputs.logits).detach().cpu().numpy()
    # get the probaility of classes
    # 0 ngeative (neutral)
    # 1 positive (hate, offense, attack)
    return softmaxed[0]