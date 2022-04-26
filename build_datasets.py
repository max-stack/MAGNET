# @Author: Max Wilson-Hebben

import json
import re
from tf_idf import PreprocessText, TF_IDF

DOLPHIN_PATH = "data/datasets/Dolphin.txt"
MAWPS_TRAIN_PATH = "data/datasets/mawps_trainset.json"
MAWPS_TEST_PATH = "data/datasets/mawps_testset.json"
MAWPS_VALID_PATH = "data/datasets/mawps_validset.json"

MAKE_CHINESE_TRAIN_PATH = "data/datasets/make_chinese_trainset.json"
MAKE_CHINESE_TEST_PATH = "data/datasets/make_chinese_testset.json"
MAKE_CHINESE_VALID_PATH = "data/datasets/make_chinese_validset.json"

EQU_DATA = "data/train/train.equ.txt"
NL_DATA = "data/train/train.nl.txt"
LDA_DATA = "data/train/train.lda.txt"

EQU_DATA_TEST = "data/test/test.equ.txt"
NL_DATA_TEST = "data/test/test.nl.txt"
LDA_DATA_TEST = "data/test/test.lda.txt"

EQU_DATA_DEV = "data/test/dev.equ.txt"
NL_DATA_DEV = "data/test/dev.nl.txt"
LDA_DATA_DEV = "data/test/dev.lda.txt"

# Used to preprocess MaKE chinese dataset for use with MAGNET
class BuildFromMakeChinese:
    def __init__(self, train_data_path, test_data_path, valid_data_path, equ_train_data, nl_train_data, lda_train_data, equ_test_data, nl_test_data, lda_test_data, equ_dev_data, nl_dev_data, lda_dev_data):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.valid_data_path = valid_data_path
        self.equ_train_data = equ_train_data
        self.nl_train_data = nl_train_data
        self.lda_train_data = lda_train_data
        self.equ_test_data = equ_test_data
        self.nl_test_data = nl_test_data
        self.lda_test_data = lda_test_data
        self.equ_dev_data = equ_dev_data
        self.nl_dev_data = nl_dev_data
        self.lda_dev_data = lda_dev_data
    
    def build_datasets(self):
        self.build_nl_equ(self.train_data_path, self.equ_train_data, self.nl_train_data)
        self.build_lda(self.nl_train_data, self.lda_train_data)
        self.build_nl_equ(self.test_data_path, self.equ_test_data, self.nl_test_data, True)
        self.build_lda(self.nl_test_data, self.lda_test_data)
        self.build_nl_equ(self.valid_data_path, self.equ_dev_data, self.nl_dev_data)
        self.build_lda(self.nl_dev_data, self.lda_dev_data)
    
    def build_dev_nums(self):
        with open("data/test/output.txt", encoding="utf8") as file:
            problem_data = file.readlines()

        with open("data/test/dev_nums.txt", encoding="utf8") as file:
            num_data = file.readlines()

        with open(self.test_data_path, encoding="utf8") as json_file:
            data = json.load(json_file)

        new_valid_set = []

        for i in range(len(problem_data)):
            problem_nums = []
            new_problem_text = ""
            new_problem = {}

            for num in num_data[i].split():
                problem_nums.append(num)

            seg = problem_data[i].split()
            for word_index in range(len(seg)):
                if seg[word_index][0] == "[" and seg[word_index][-1] == "]":
                    new_problem_text += problem_nums[int(seg[word_index][-2])-1] + " "
                else:
                    new_problem_text += seg[word_index] + " "
                
            new_problem["id"] = data[i]["id"]
            new_problem["original_text"] = new_problem_text
            new_problem["segmented_text"] = new_problem_text
            new_problem["equation"] = data[i]["equation"]
            new_problem["ans"] = data[i]["ans"]

            new_valid_set.append(new_problem)
        
        with open("data/test/output_test.txt", "w", encoding="utf-8") as output_file:
            json.dump(new_valid_set, output_file, ensure_ascii=False, indent=4)
        
    def build_nl_equ(self, data_path, equ_data, nl_data, dev=False):
        with open(data_path, encoding="utf8") as json_file:
            data = json.load(json_file)
        
        equ_file = open(equ_data, "w", encoding="utf-8")
        nl_file = open(nl_data, "w", encoding="utf-8")

        if dev:
            dev_nums_file = open("data/test/dev_nums.txt", "w", encoding="utf-8")

        for problem in data:
            nl_text = problem["segmented_text"]
            equation_text = problem["equation"]
            equation_text = re.sub('([^,.\d])', r' \1 ', equation_text)
            equation_text = equation_text.replace("  ", " ")
            equation_text = equation_text.replace(";", "EOE")
            texts = self.add_nums(equation_text, nl_text)
            equ_file.write(texts[0] + "\n")
            nl_file.write(texts[1] + "\n")
            if dev:
                for num in texts[2]:
                    dev_nums_file.write(num + " ")
                dev_nums_file.write("\n")
        
        nl_file.close()
        equ_file.close()

    def add_nums(self, equation_text, nl_text):
        curNum = ""
        curNums = []
        new_equation_text = ""
        curIndex = 0

        for i in range(len(equation_text)):
            char = equation_text[i]
            if char.isdigit() or char == '.':
                curNum += char
            else: 
                if curNum != "":
                    for j in range(len(curNums)):
                        if curNum == curNums[j]:
                            new_equation_text += "[num" + str(j) + "]"
                    
                    if len(new_equation_text) == 0 or not new_equation_text[-1] == "]":
                        new_equation_text += "[num" + str(curIndex) + "]"
                        curIndex += 1
                        curNums.append(curNum)
                    
                    curNum = ""

                new_equation_text += char
                
        if curNum != "":
            for j in range(len(curNums)):
                if curNum == curNums[j]:
                    new_equation_text += "[num" + str(j) + "]"
            
            if not new_equation_text[-1].isdigit():
                new_equation_text += "[num" + str(curIndex) + "]"
                curIndex += 1
                curNums.append(curNum)
                    
            curNum = ""

        new_nl_text = ""

        for word in nl_text.split():
            if word[0].isdigit():
                for num in range(len(curNums)):
                    if word == curNums[num] or word + ".0" == curNums[num]:
                        new_nl_text += "[num" + str(num) + "] "
                        break
            else:
                new_nl_text += word + " "

        return [new_equation_text + " EOE", new_nl_text, curNums]
            
    
    def build_lda(self, nl_data, lda_data):
        data = []
        nl_file = open(nl_data, 'r', encoding='utf-8')
        lines = nl_file.readlines()

        nl_file.close()

        for line in lines:
            data.append(line)

        processed_text = PreprocessText(data)
        processed_text.preprocess_chinese()

        tf_idf = TF_IDF(processed_text.original_text)
        tf_idf.calculate_tf_idf()

        lda_file = open(lda_data, "w", encoding="utf-8")

        for document in tf_idf.tf_idf:
            for word in document:
                lda_file.write(word + " ")
            lda_file.write("\n")
        
        lda_file.close()


# Used to preprocess MAWPS chinese dataset for use with MAGNET
class BuildFromMawps:
    def __init__(self, train_data_path, test_data_path, valid_data_path, equ_train_data, nl_train_data, lda_train_data, equ_test_data, nl_test_data, lda_test_data, equ_dev_data, nl_dev_data, lda_dev_data):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.valid_data_path = valid_data_path
        self.equ_train_data = equ_train_data
        self.nl_train_data = nl_train_data
        self.lda_train_data = lda_train_data
        self.equ_test_data = equ_test_data
        self.nl_test_data = nl_test_data
        self.lda_test_data = lda_test_data
        self.equ_dev_data = equ_dev_data
        self.nl_dev_data = nl_dev_data
        self.lda_dev_data = lda_dev_data
    
    def build_datasets(self):
        self.build_nl_equ(self.train_data_path, self.equ_train_data, self.nl_train_data)
        self.build_lda(self.nl_train_data, self.lda_train_data)
        self.build_nl_equ(self.test_data_path, self.equ_test_data, self.nl_test_data)
        self.build_lda(self.nl_test_data, self.lda_test_data)
        self.build_nl_equ(self.valid_data_path, self.equ_dev_data, self.nl_dev_data, True)
        self.build_lda(self.nl_dev_data, self.lda_dev_data)
    
    def build_dev_nums(self):
        with open("data/test/output.txt", encoding="utf8") as file:
            problem_data = file.readlines()

        with open("data/test/dev_nums.txt", encoding="utf8") as file:
            num_data = file.readlines()

        with open(self.valid_data_path, encoding="utf8") as json_file:
            data = json.load(json_file)

        new_valid_set = []

        for i in range(len(problem_data)):
            problem_nums = []
            new_problem_text = ""
            new_problem = {}

            for num in num_data[i].split():
                problem_nums.append(num)

            seg = problem_data[i].split()
            for word_index in range(len(seg)):
                if seg[word_index][0] == "[" and seg[word_index][-1] == "]":
                    new_problem_text += problem_nums[int(seg[word_index][-2])-1] + " "
                else:
                    new_problem_text += seg[word_index] + " "
                
            new_problem["id"] = data[i]["id"]
            new_problem["original_text"] = new_problem_text
            new_problem["segmented_text"] = new_problem_text
            new_problem["equation"] = data[i]["equation"]
            new_problem["ans"] = data[i]["ans"]

            new_valid_set.append(new_problem)
        
        with open("data/test/output_test.txt", "w", encoding="utf-8") as output_file:
            json.dump(new_valid_set, output_file, ensure_ascii=False, indent=4)

    def build_nl_equ(self, data_path, equ_data, nl_data, dev=False):
        with open(data_path, encoding="utf8") as json_file:
            data = json.load(json_file)
        
        equ_file = open(equ_data, "w", encoding="utf-8")
        nl_file = open(nl_data, "w", encoding="utf-8")

        if dev:
            dev_nums_file = open("data/test/dev_nums.txt", "w", encoding="utf-8")

        for problem in data:
            nl_text = problem["original_text"]
            equation_text = problem["equation"].lower()
            equation_text = re.sub('([^,.\d])', r' \1 ', equation_text)
            equation_text = equation_text.replace("  ", " ")
            texts = self.add_nums(equation_text, nl_text)
            equ_file.write(texts[0] + "\n")
            nl_file.write(texts[1] + "\n")
            if dev:
                for num in texts[2]:
                    dev_nums_file.write(num + " ")
                dev_nums_file.write("\n")
        
        nl_file.close()
        equ_file.close()
    
    def add_nums(self, equation_text, nl_text):
        curNum = ""
        curNums = []
        new_equation_text = ""
        curIndex = 0

        for i in range(len(equation_text)):
            char = equation_text[i]
            if char.isdigit() or char == '.':
                curNum += char
            else: 
                if curNum != "":
                    for j in range(len(curNums)):
                        if curNum == curNums[j]:
                            new_equation_text += "[num" + str(j) + "]"
                    
                    if not len(new_equation_text) == 0 and not new_equation_text[-1].isdigit():
                        new_equation_text += "[num" + str(curIndex) + "]"
                        curIndex += 1
                        curNums.append(curNum)
                    
                    curNum = ""

                new_equation_text += char
                
        if curNum != "":
            for j in range(len(curNums)):
                if curNum == curNums[j]:
                    new_equation_text += "[num" + str(j) + "]"
            
            if not new_equation_text[-1].isdigit():
                new_equation_text += "[num" + str(curIndex) + "]"
                curIndex += 1
                curNums.append(curNum)
                    
            curNum = ""

        new_nl_text = ""

        for word in nl_text.split():
            if word[0].isdigit():
                for num in range(len(curNums)):
                    if word == curNums[num] or word + ".0" == curNums[num]:
                        new_nl_text += "[num" + str(num) + "] "
                        break
            else:
                new_nl_text += word + " "
        
        if new_equation_text[-1] == " ":
            new_equation_text += "EOE"
        else:
            new_equation_text += " EOE"
        return [new_equation_text, new_nl_text, curNums]
            
    
    def build_lda(self, nl_data, lda_data):
        data = []
        nl_file = open(nl_data, 'r', encoding='utf-8')
        lines = nl_file.readlines()

        nl_file.close()

        for line in lines:
            data.append(line)

        processed_text = PreprocessText(data)
        processed_text.preprocess()

        tf_idf = TF_IDF(processed_text.original_text)
        tf_idf.calculate_tf_idf()

        lda_file = open(lda_data, "w", encoding="utf-8")

        for document in tf_idf.tf_idf:
            for word in document:
                lda_file.write(word + " ")
            lda_file.write("\n")
        
        lda_file.close()

processed_data = BuildFromMawps(MAWPS_TRAIN_PATH, MAWPS_TEST_PATH, MAWPS_VALID_PATH, EQU_DATA, NL_DATA, LDA_DATA, EQU_DATA_TEST, NL_DATA_TEST, LDA_DATA_TEST, EQU_DATA_DEV, NL_DATA_DEV, LDA_DATA_DEV)
processed_data.build_datasets()
processed_data.build_dev_nums()

    
# removes problems with different number of [numX] values in equation to the MWP.
def remove_different_nums(equ_data, nl_data, lda_data):
    equ_nums_used = []
    nl_nums_used = []
    nl_file = open(nl_data, 'r', encoding='utf-8')
    equ_file = open(equ_data, 'r', encoding='utf-8')

    nl_lines = nl_file.readlines()
    equ_lines = equ_file.readlines()

    nl_file.close()
    equ_file.close()
    line = -1
    while line < len(equ_lines):
        line += 1
        if(line >= len(equ_lines)):
            break
        equ_nums_used.clear()
        nl_nums_used.clear()
        for num in re.findall(r'\[([A-Za-z0-9_]+)\]', equ_lines[line]):
            equ_nums_used.append(num)
    
        for num in re.findall(r'\[([A-Za-z0-9_]+)\]', nl_lines[line]):
            nl_nums_used.append(num)

        deleted = False
        for equ_num in equ_nums_used:
            if not equ_num in nl_nums_used:
                del equ_lines[line]
                del nl_lines[line]
                deleted = True
                line -= 1
                break

        if deleted:
            continue

        for nl_num in nl_nums_used:
            if not nl_num in equ_nums_used:
                del equ_lines[line]
                del nl_lines[line]
                line -= 1
                break      
        

    nl_file = open(nl_data, 'w+', encoding='utf-8')
    equ_file = open(equ_data, 'w+', encoding='utf-8')

    for line in range(len(equ_lines)):
        equ_file.write(equ_lines[line])
        nl_file.write(nl_lines[line])

# remove_different_nums(EQU_DATA_TEST, NL_DATA_TEST, LDA_DATA_TEST)


# used to find topic words from Dolphin18K datset and process data.
class BuildFromDolphin:
    def __init__(self, data_path, equ_data, nl_data, lda_data):
        self.data_path = data_path
        self.equ_data = equ_data
        self.nl_data = nl_data
        self.lda_data = lda_data

    def build_datasets(self):
        self.build_nl_equ()
        self.build_lda()
        self.remove_empty_entries()

    def build_nl_equ(self):
        with open(self.data_path, encoding="utf8") as json_file:
            data = json.load(json_file)

        equ_file = open(self.equ_data, "w", encoding="utf-8")
        nl_file = open(self.nl_data, "w", encoding="utf-8")

        for problem in data:
            nl_file.write(problem["text"].partition("\"")[
                0].replace("\n", "").replace("\r", "") + "\n")

            equations = re.findall(
                "(?<=equ:)(.*?)(?=\n|\r|$)", problem["equations"])
            equation_set = ""

            for equation in equations:
                equation = equation.replace(" ", "")
                equation_set += equation + " "

            equ_file.write(equation_set + "\n")

        equ_file.close()
        nl_file.close()

    def build_lda(self):
        data = []
        nl_file = open(self.nl_data, 'r', encoding='utf-8')
        lines = nl_file.readlines()

        nl_file.close()

        for line in lines:
            data.append(line)

        processed_text = PreprocessText(data)
        processed_text.preprocess()

        tf_idf = TF_IDF(processed_text.original_text)
        tf_idf.calculate_tf_idf()

        lda_file = open(self.lda_data, "w", encoding="utf-8")

        for document in tf_idf.tf_idf:
            for word in document:
                lda_file.write(word + " ")
            lda_file.write("\n")

    def remove_empty_entries(self):
        nl_file = open(self.nl_data, 'r', encoding='utf-8')
        equ_file = open(self.equ_data, 'r', encoding='utf-8')
        lda_file = open(self.lda_data, 'r', encoding='utf-8')

        nl_lines = nl_file.readlines()
        equ_lines = equ_file.readlines()
        lda_lines = lda_file.readlines()

        nl_file = open(self.nl_data, 'w', encoding='utf-8')
        equ_file = open(self.equ_data, 'w', encoding='utf-8')
        lda_file = open(self.lda_data, 'w', encoding='utf-8')

        line = 0
        while line < len(nl_lines):
            if not(nl_lines[line] == "\n" or equ_lines[line] == "\n" or lda_lines[line] == "\n"):
                nl_file.write(nl_lines[line])
                equ_file.write(equ_lines[line])
                lda_file.write(lda_lines[line])
                line += 1
            line += 1

        nl_file.close()
        equ_file.close()
        lda_file.close()

