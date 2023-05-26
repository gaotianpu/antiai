# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document
import sys 
import requests
import random
import json
from hashlib import md5
from config import trans_appid,trans_appkey

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'en'
to_lang =  'zh'

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

def trans(query):
    salt = random.randint(32768, 65536)
    sign = make_md5(trans_appid + query + str(salt) + trans_appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': trans_appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    
    segments = result.get("trans_result")
    for segment in segments:
        dst = segment.get("dst")
        if dst.startswith("#"):
            print(segment.get("src") + " " + dst.replace("#",""))
        elif dst.startswith("!["):
            print(segment.get("src"))
        else:    
            print(segment.get("src") + "\n")
            print(segment.get("dst") + "\n") 

def process(filename):
    with open(filename) as f :
        lines = f.readlines()
        total_len = 0
        parts = []
        part_idx = 0
        for i,line in enumerate(lines):
            if total_len + len(line) > 6000:
                query = "\n".join(parts)
                trans(query)
                # print(part_idx,query[:100].replace("\n",""))
                
                part_idx = part_idx + 1
                total_len = 0
                parts = []
            
            parts.append(line)
            total_len = total_len + len(line)
        
        query = "\n".join(parts)
        part_idx = part_idx + 1
        trans(query)
        # print(part_idx,query[:100].replace("\n",""))

if __name__ == "__main__":
    process(sys.argv[1])
