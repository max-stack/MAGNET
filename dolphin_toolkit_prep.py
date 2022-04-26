# @Author: Max Wilson-Hebben

import json

NL_TRAIN_PATH = "data/train/train.nl.txt"
NL_TEST_PATH = "data/test/test.nl.txt"
NL_DEV_PATH = "data/test/dev.nl.txt"

EQU_TRAIN_PATH = "data/train/train.equ.txt"
EQU_TEST_PATH = "data/test/test.equ.txt"
EQU_DEV_PATH = "data/test/dev.equ.txt"

TRAIN_TEMPLATE_PATH = "train_template_data.json"
TEST_TEMPLATE_PATH = "test_template_data.json"
VALID_TEMPLATE_PATH = "valid_template_data.json"

# used to preprocess original Dolphin18K dataset
class dolphin_toolkit_prep:
    def __init__(self, nl_train_path, nl_test_path, nl_dev_path, equ_train_path, equ_test_path, equ_dev_path, train_template_path, test_template_path, valid_template_path):
        self.nl_train_path = nl_train_path
        self.nl_test_path = nl_test_path
        self.nl_dev_path = nl_dev_path

        self.equ_train_path = equ_train_path
        self.equ_test_path = equ_test_path
        self.equ_dev_path = equ_dev_path

        self.train_template_path = train_template_path
        self.test_template_path = test_template_path
        self.valid_template_path = valid_template_path

    def train_generate_json(self):
        question_list = []
        
        equ_file = open(self.equ_train_path, "r", encoding="utf-8")
        nl_file = open(self.nl_train_path, "r", encoding="utf-8")

        equ_lines = equ_file.readlines()
        nl_lines = nl_file.readlines()

        id = 0

        for i in range(len(nl_lines)):
            question = {}
            question["id"] = id
            question["original_text"] = nl_lines[i]
            
            equation = self.adjust_equation(equ_lines[i])
            question["equation"] = equation
            question["ans"] = 0
            question_list.append(question)
            id += 1
        
        with open(self.train_template_path, "w", encoding="utf-8") as file:
            json.dump(question_list, file, ensure_ascii=False, indent=4)

    def test_generate_json(self):
        question_list = []
        
        equ_file = open(self.equ_test_path, "r", encoding="utf-8")
        nl_file = open(self.nl_test_path, "r", encoding="utf-8")

        equ_lines = equ_file.readlines()
        nl_lines = nl_file.readlines()

        id = 0

        for i in range(len(nl_lines)):
            question = {}
            question["id"] = id
            question["original_text"] = nl_lines[i]
            
            equation = self.adjust_equation(equ_lines[i])
            question["equation"] = equation
            question["ans"] = 0
            question_list.append(question)
            id += 1
        
        with open(self.test_template_path, "w", encoding="utf-8") as file:
            json.dump(question_list, file, ensure_ascii=False, indent=4)

    def valid_generate_json(self):
        question_list = []
        
        equ_file = open(self.equ_dev_path, "r", encoding="utf-8")
        nl_file = open(self.nl_dev_path, "r", encoding="utf-8")

        equ_lines = equ_file.readlines()
        nl_lines = nl_file.readlines()

        id = 0

        for i in range(len(nl_lines)):
            question = {}
            question["id"] = id
            question["original_text"] = nl_lines[i]
            
            equation = self.adjust_equation(equ_lines[i])
            question["equation"] = equation
            question["ans"] = 0
            question_list.append(question)
            id += 1
        
        with open(self.valid_template_path, "w", encoding="utf-8") as file:
            json.dump(question_list, file, ensure_ascii=False, indent=4)
    
    def adjust_equation(self, equation):
        equation = equation.replace(" ", "")
        equation = equation.replace("EOE", " ; ")
        equation = equation.rsplit(";", 1)[0]
        equation = equation.strip()
        return equation

test = dolphin_toolkit_prep(NL_TRAIN_PATH, NL_TEST_PATH, NL_DEV_PATH, EQU_TRAIN_PATH, EQU_TEST_PATH, EQU_DEV_PATH, TRAIN_TEMPLATE_PATH, TEST_TEMPLATE_PATH, VALID_TEMPLATE_PATH)
test.train_generate_json()
test.test_generate_json()
test.valid_generate_json()
         

    
