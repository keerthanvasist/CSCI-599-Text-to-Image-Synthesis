import os
import random
import numpy as np


sentenceFile = open("sentences-nlp.txt", 'r')

trainingSetPath = 'Data/trainingSet/'
inputPath = r'./dataset/input/'
sentenceTypes = {}
numExamples = 30000



def fetchSentenceTemplates():
    line = sentenceFile.readline()
    action = ""
    while line != "":
        splits = line.split(' ')
        if splits[0] == '****':
            action = splits[1]
            sentenceTypes[action] = []
        else :
            sentencelist = sentenceTypes.get(action)
            sentencelist.append(line)
        line = sentenceFile.readline()
    return

def getSolution(actions, number1, number2):
    if actions.strip() == 'Multiply':
        answer = number2*number1
        return str(answer)
    elif actions.strip() == 'Square':
        answer = number1*number1
        return str(answer)
    elif actions.strip() == 'Add':
        answer = number1 + number2
        return str(answer)

def getrandomSentence(sentenceType,number1, number2):
    sentences = sentenceTypes.get(sentenceType)
    randomNumber = random.randint(0,len(sentences)-1)
    sentence = sentences[randomNumber]
    sentence = sentence.replace('{number1}',str(number1))
    sentence = sentence.replace('{number2}',str(number2))
    return sentence

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    fetchSentenceTemplates()
    actions = []
    for sentenceType in sentenceTypes:
        actions.append(sentenceType)
    sentences = ''
    print actions
    for i in range(numExamples):
        action = actions[random.randint(0,2)]
        number1 = random.randint(0,9)
        number2 = random.randint(0, 9)
        sentences = sentences + getrandomSentence(action, number1, number2).strip()+' ' + getSolution(action,number1, number2) + '\n'
    #print(sentences)
    sentence_file = open("sentences_nlp_test.txt", 'w+')
    sentence_file.write(sentences)
    sentence_file.close()