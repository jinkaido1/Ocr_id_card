# Ocr_id_card
用tesseract 进行身份证识别

参考了该项目，学习了身份证识别的流程，看起来有点乱，重新对其做了一番整理。
ref:https://github.com/sam-ke/idCardOcr

流程大致是：先找到身份证号码区域，在根据号码区域来找到各个其它信息区域，然后将每个区域的原图或者处理的图用tesseract工具进行识别。

效果并不是很好。汉字部分识别较弱，需要不断尝试二值化阈值参数及其它参数。

# Tools
首先下载tesseract-ocr工具，wins下载地址:https://digi.bib.uni-mannheim.de/tesseract/ 
运行安装(安装过程勾选语言包)。记住安装目录（一般是 C:\Program Files (x86)\Tesseract-OCR\tesseract.exe）

- python3

- pip install pytesseract

程序中设置：pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'，程序pytesseract将会调用该工具进行识别。

# Results

```
身份证有效
44030119840217411X
{'CARD_ETHNIC': '汉', 'CARD_MON': '', 'CARD_ADDR': '村115栋21A', 'CARD_YEAR': '', 'CARD_NAME': '男民族汉', 'CARD_DAY': '', 'CARD_SEX': '984'}
```
