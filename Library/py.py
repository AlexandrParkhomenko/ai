def file_read(file_name):
    # читаем обычный файл патча
    file_patch = open(file_name, "r")
    patch = file_patch.read()
    file_patch.close()
    return patch

def file_write(file_name, file_content):
    file_patch = open(file_name, "w")
    file_patch.write(file_content)
    file_patch.close()
    return

import sys
print(sys.version)

#import re

f1 = "/home/a/Stanford/cs224n/subtitles/" + "1 - Introduction and Word Vectors.ru.srt"
subtitles = file_read(f1).split("\n")
#subtitles = re.split(r'\n{2,3}', subtitles)
#for sub in subtitles:
i = 0
step = 4
while i < len(subtitles)-step:
    #print(subtitles[i+2])
    if subtitles[i+2][0:2] == ", ":
        subtitles[i+2-4] += 
        subtitles[i+2] = subtitles[i+2][2:] # without comma
        print(subtitles[i+2][2:])
    i += step
