# -*- coding: utf-8 -*-
import sys


# 用于转化某些固定术语

def load_dict():
    d = {}
    with open('translate.dict') as f:
        for i,line in  enumerate(f):
            if i<2: continue 
            parts = line.strip().split("|")
            assert len(parts) == 3
            d[parts[1].strip()] = parts[2].strip()
    return d 


if __name__ == "__main__":
    d = load_dict()
    for line in sys.stdin:   
        for k,v in d.items():
            line = line.strip().replace(k,v)
        print(line)