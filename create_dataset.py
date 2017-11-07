import os
import random
import cv2
import numpy as np


sentenceFile = open("sentences.txt", 'r')

trainingSetPath = 'Data/trainingSet/'
inputPath = r'./dataset/input/'
sentenceTypes = {}
numExamples = 600;


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


def getSolution(actions, number, number1):
    if actions.strip() == 'Multiply':
        answer = number*number1
        return str(answer)
    elif actions.strip() == 'Square':
        answer = number*number
        return str(answer)
    

def getFilenames():
    labelDirs = next(os.walk(trainingSetPath))[1]    
    filenames = []
    for dir in labelDirs:
        path = trainingSetPath+str(dir)
        filenames.append(os.listdir(path))
    return filenames


def getrandomSentence(sentenceType,number1):
    sentences = sentenceTypes.get(sentenceType)
    randomNumber = random.randint(0,len(sentences)-1)
    sentence = sentences[randomNumber]
    sentence = sentence.replace('{number1}',str(number1))
    return sentence

def getRandomImage(number, filenames):
    randomNumber = random.randint(0,len(filenames[number])-1)
    imagePath = trainingSetPath+str(number)+'/'+filenames[number][randomNumber]
    return cv2.imread(imagePath.encode(),0)

def create_GAN_dataset():
    print('Creating GAN Dataset...')
    filenames = getFilenames()
    fetchSentenceTemplates()
    index = 0;
    input_sentences = []
    for sentenceType in sentenceTypes:
        for i in range(numExamples):
            number = random.randint(0,9)
            number1 = random.randint(0,9)
            
            inputSentence = getrandomSentence(sentenceType, number1)
            answer = getSolution(sentenceType, number,number1)
            inputSentence = inputSentence.strip() +" "+str(answer)
            input_sentences.append(inputSentence)
            inputImage =  getRandomImage(number, filenames)
            pic_name = r'./dataset/input/img_'+str(index)+'.jpg'
            cv2.imwrite(pic_name, inputImage )
            
            if len(answer) == 1:
                pic_name = r'./dataset/output/img_'+str(index)+'.jpg'
                ensure_dir(pic_name)
                cv2.imwrite(pic_name, getRandomImage(int(answer),filenames))
            else:
                number1 = int(answer[0])
                number2 = int(answer[1])
                
                input1 = getRandomImage(number1, filenames)
                input2 = getRandomImage(number2, filenames)
                
                output_image = concatenateAndResize(input1, input2)
                pic_name = r'./dataset/output/img_'+str(index)+'.jpg'
                ensure_dir(pic_name)
                cv2.imwrite(pic_name, output_image )
                
            
            index += 1
            
            
    ensure_dir("dataset/input/sentences.txt")
    sentence_file = open("dataset/input/sentences.txt",'w+')
    
    for sentence in input_sentences:
        sentence_file.write(sentence+'\n')
    sentence_file.close()
    
    return

def createMNIST100():

    print('Creating MNIST100 Dataset...')
    size = 600
    filenames = getFilenames()
    for i in range(100):
        print ("Generating "+str(i))
        for j in range(size):
            num = str(i)
            image_name = r'dataset/mnist/'+num+'/img_'+str(j)+'.jpg'
            ensure_dir(image_name)
            
            if i > 9:
                number1 = int(num[0])
                number2 = int(num[1])
            
                input1 = getRandomImage(number1, filenames)
                input2 = getRandomImage(number2, filenames)
                
                output_image = concatenateAndResize(input1, input2)
                cv2.imwrite(image_name, output_image )
            else :
                output_image = getRandomImage(i, filenames)
                cv2.imwrite(image_name, output_image )
                
    return

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def concatenateAndResize(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    #create empty matrix
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    
    #combine 2 images
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    resized_image = cv2.resize(vis, (28, 28)) 
    
    return resized_image


if __name__ == "__main__":
    createGANDataset = False
    
    if createGANDataset :
        create_GAN_dataset()
    else : 
        createMNIST100()
