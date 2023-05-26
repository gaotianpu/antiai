# 工具集

* pdf2md.py 

经来自pdf的复制粘贴文本格式化为markdown格式的文档。 段落处理，加#号等还有不少问题

cat pdf.txt | python pdf2md.py > pdf.md


* translate.py 

调用百度的翻译API,将英文翻译成中文

python translate.py pdf.md > cn_pdf.md


* trans_dict.py 

一些固定的容易翻译错误的术语纠正, 结合translate.dict文件使用。

cat cn_pdf.md | python trans_dict.py > cn_fixed_pdf.md


* pdf2txt
https://github.com/jsvine/pdfplumber

pdfplumber < /mnt/d/paper/vit_JPEG.2211.16421.pdf > pdf.csv