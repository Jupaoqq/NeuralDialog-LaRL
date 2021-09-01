import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


def process(input, output, dummy = True):
    f = open(input, encoding='utf-8')
    f2 = open(output, "a", encoding='utf-8')
    for line in tqdm(f):
        lines=json.loads(line.strip())
        seekerid=lines["initiatorWorkerId"]
        recommenderid=lines["respondentWorkerId"]
        contexts=lines['messages']
        movies=lines['movieMentions']
        altitude=lines['respondentQuestions']
        initial_altitude=lines['initiatorQuestions']
        l = ""
        if (altitude and initial_altitude):
            if(dummy):
                l += "<input> 1 1 1 1 1 1 </input> " 
            # else:
            #     l += "<input> "
            #     for key, a in initial_altitude.items():
            #         l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            #     l += "</input> "
            last = ""
            l += "<dialogue> "
            temp_id = 100000000000
            for m in contexts:
                if m['senderWorkerId'] != temp_id:
                    if temp_id != 100000000000:
                        l += " <eos> "
                    temp_id = m['senderWorkerId']
                    if m['senderWorkerId'] == seekerid:
                        l += "YOU: "
                        last = "THEM: <selection> "
                    elif m['senderWorkerId'] == recommenderid:
                        l += "THEM: "
                        last = "YOU: <selection> "
                    else:
                        pass
                l += m['text']      
                l += " "
            l += " <eos> "
            l += last
            l += "</dialogue> " 
            if(dummy):
                l += "<output> item0=1 item1=1 item2=1 item0=1 item1=1 item2=1 </output> <partner_input> 1 1 1 1 1 1 </partner_input>"
            # else:
            #     l += "<output> "
            #     count1 = 0
            #     for a in initial_altitude.values():
            #         l += ("item%d=" % (count1))
            #         l += str(a['suggested']) + " "
            #         count1 += 1
            #     count2 = 0
            #     for a in altitude.values():
            #         l += ("item%d=" % (count2))
            #         l += str(a['liked']) + " "
            #         count2 += 1
            #     l += "</output> "
            #     l += "<partner_input> "
            #     for key, a in altitude.items():
            #         l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            #     l += "</partner_input> "
            l += " <user> "
            for key, a in altitude.items():
                l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            l += "</user> "
            l = l.replace('\r', '')
            l = l.replace('\n', '')
            l += "\n"
            # print(l)
            f2.write(l)

def movie(input, output):
    f = open(input, encoding='utf-8')
    f2 = open(output, "a", encoding='utf-8')
    for line in tqdm(f):
        lines=json.loads(line.strip())
        initial_altitude=lines['initiatorQuestions']
        if initial_altitude: 
            l = "1 1 1 1 1 1 " 
            for key, a in initial_altitude.items():
                l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            l += "\n"
            # print(l)
            f2.write(l)

def entity(input1, input2, input3, output):
    f1 = open(input1, encoding='utf-8')
    f2 = open(input2, encoding='utf-8')
    f3 = open(input3, encoding='utf-8')
    f4 = open(output, "a", encoding='utf-8')
    temp = []

    for line in tqdm(f1):
        lines=json.loads(line.strip())
        initial_altitude=lines['initiatorQuestions']
        if initial_altitude: 
            for key, a in initial_altitude.items():
                if key not in temp:
                    temp.append(key)
    for line in tqdm(f2):
        lines=json.loads(line.strip())
        initial_altitude=lines['initiatorQuestions']
        if initial_altitude: 
            for key, a in initial_altitude.items():
                if key not in temp:
                    temp.append(key)
    for line in tqdm(f3):
        lines=json.loads(line.strip())
        initial_altitude=lines['initiatorQuestions']
        if initial_altitude: 
            for key, a in initial_altitude.items():
                if key not in temp:
                    temp.append(key)

    for t in temp:
        f4.write(t + "\n")

if __name__=='__main__':
    process('data/raw/train_data.jsonl', 'data/negotiate/train.txt')
    process('data/raw/valid_data.jsonl', 'data/negotiate/val.txt')
    process('data/raw/test_data.jsonl', 'data/negotiate/test.txt')
    movie('data/raw/train_data.jsonl', 'data/negotiate/selfplay.txt')
    movie('data/raw/valid_data.jsonl', 'data/negotiate/selfplay_eval.txt')
    # entity('data/raw/train_data.jsonl', 'data/raw/valid_data.jsonl', 'data/raw/test_data.jsonl', 'data/negotiate/entity.txt')