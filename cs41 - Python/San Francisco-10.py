
"""
@author: AlexandrParkhomenko

Определите, сколько раз во всём рассказе И. А. Бунина «Господин из Сан-Франциско» встречается сочетание букв «из» или «Из» только в составе других слов (например, в словах «низко» или «из-за»), но не как отдельное слово.
# Доступны файлы для чтения: 10.doc, 10.docx, 10.odt, 10.pdf, 10.txt
"""
f = open('10.txt', 'r')
c = f.read() #" Из из из-за" #
#print(c)
import re
x = re.findall(r'(\wиз\W|\Wиз\w|\wиз\w|\Wиз-\w)', c, re.I)
print(len(x)) #62

