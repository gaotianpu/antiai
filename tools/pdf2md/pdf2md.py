# coding: utf-8
import sys
import re 

p = re.compile(r"[\d]+[\.]+.")

p1 = re.compile(r"^[A-Z]+[\.]+[\d]? ")
p2 = re.compile(r"^[A-Z]+[\.]+[\d][\.]? ")

aft_References = False
last_line = ""
for line in sys.stdin:   
    line =  line.strip().replace("￾","")

    if line.lower() == "references":
        line = "## References"
        aft_References = True
    
    if line.lower().startswith("appendix"):
        aft_References = False

    if p.match(line) and not aft_References:  
        dot_cnt = line.count(".")
        str_header = ""
        for i in range(0,dot_cnt):
            str_header = str_header + "#"
        line = "#" + str_header + " " + line 
    
    if (p1.match(line) or p2.match(line)) and not aft_References:  
        dot_cnt = line.count(".")
        str_header = ""
        for i in range(0,dot_cnt):
            str_header = str_header + "#"
        line = "##" + str_header + " " + line 

    if line.lower()  == "abstract":
        line = "## Abstract"
    
    if line.lower() == "1 introduction":
        line = "## 1 Introduction" 
    
    if line.lower() == "2 related work":
        line = "## 2 Related Work" 
    
    if line.lower() == "3 method":
        line = "## 3 Method" 
    
    if line.lower() == "4 implementation":
        line = "## 4 Implementation" 
    
    if line.lower() == "acknowledgments":
        line = "## Acknowledgments" 
    
    if line.startswith("• "):
        line = line.replace("• ","\n* ")
    
    if line.startswith("#"):
        if last_line:
            print(last_line+"\n")
        print(line) 
        last_line = ""
        continue 
    
    if aft_References : 
        if line.startswith("["):
            print(last_line) 
            last_line = line.replace("[","").replace(']','.') 
        else :
            last_line = last_line + " " + line 
        continue 
    
    if line and line[0].isupper():
        if last_line:
            print(last_line+"\n")
        last_line = line
    elif line and line.startswith("• "):
        # if last_line:
        print(last_line.replace("• ","* "))
        last_line = line
    else :
        last_line = last_line + " " + line
    
    

print(last_line) 

# \n\n^Inception  ->  Inception
# \n\n^Xception  ->  Xception

# cat test.md | python pdf2md.py > Inception_v4.md
