import json
import re

DATA_PATH = "data/Dolphin.txt"
EQU_DATA = "data/train/train.equ.txt"
NL_DATA = "data/train/train.nl.txt"

with open(DATA_PATH, encoding="utf8") as json_file:
    data = json.load(json_file)

equ_file = open(EQU_DATA, "w", encoding="utf-8")
nl_file = open(NL_DATA, "w", encoding="utf-8")

for problem in data:
    nl_file.write(problem["text"].partition("\"")[0] + "\n")

    equations = re.findall("(?<=equ:)(.*?)(?=\n|\r|$)", problem["equations"])
    equation_set = ""

    for equation in equations:
        equation = equation.replace(" ", "")
        equation_set += equation + " "

    equ_file.write(equation_set + "\n")

equ_file.close()
nl_file.close()