import os
import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

trainingSetPath = 'Data/trainingSet/'
sentenceTypes = []
sentenceTypes.append("Multiply {number1} by {number2}")
sentenceTypes.append("Square {number1}")
numExamples = 10;



def getSolution(sentence, number1, number2):
    words  = sentence.split(" ")
    if words[0] == 'Multiply':
        answer = number1*number2
        return str(answer)
    elif words[0] == 'Square':
        answer = number1*number1
        return str(answer)
    


def main():
    labelDirs = next(os.walk(trainingSetPath))[1]
    #print(labelDirs)
    filenames = []
    for dir in labelDirs:
        path = trainingSetPath+str(dir)
        filenames.append(os.listdir(path))
    
    index = 0;
    input_sentences = []
    for sentence in sentenceTypes:
        for i in range(numExamples):
            number1 = random.randint(0,9)
            number2 = random.randint(0,9)
            inputSentence = sentence.replace("{number1}",str(number1))
            inputSentence = inputSentence.replace("{number2}",str(number2))
            input_sentences.append(inputSentence)

            numImages = len(filenames[number1])
            numImages = random.randint(1,numImages-1)
            imagePath = trainingSetPath+str(number1)+'/'+filenames[number1][numImages]
            inputImage =  cv2.imread(imagePath.encode(),0)

            #cv2.imshow('image',inputImage)
            pic_name = r'./dataset/input/img_'+str(index)+'.jpg'
            print(cv2.imwrite(pic_name, inputImage ))
            
            answer = getSolution(inputSentence, number1,number2)
            print(answer)
            
            imagePath = trainingSetPath
            if len(answer) == 1:
                numImages = len(filenames[number1])
                numImages = random.randint(1,numImages-1)
                imagePath = imagePath+answer+"/"+filenames[int(answer)][numImages]
                inputImage =  cv2.imread(imagePath.encode(),0)
                
                pic_name = r'./dataset/output/img_'+str(index)+'.jpg'
                print(cv2.imwrite(pic_name, inputImage ))
            else:
                number1 = int(answer[0])
                number2 = int(answer[1])
                
                numImages = len(filenames[number1])
                numImages = random.randint(1,numImages-1)
                image1Path = imagePath+str(number1)+"/"+filenames[number1][numImages]
                input1 =  cv2.imread(image1Path.encode(),0)

                
                numImages = len(filenames[number2])
                numImages = random.randint(1,numImages-1)
                image2Path = imagePath+str(number2)+"/"+filenames[number2][numImages]
                input2 =  cv2.imread(image2Path.encode(),0)
                
                output_image = concatenateAndResize(input1, input2)
                pic_name = r'./dataset/output/img_'+str(index)+'.jpg'
                print(cv2.imwrite(pic_name, output_image ))
                
            
            index += 1
            
    
    sentence_file = open("dataset/input/sentences.txt",'w+')
    
    for sentence in input_sentences:
        sentence_file.write(sentence+'\n')
    sentence_file.close()
            

    combined_image = concatenateAndResize(cv2.imread("Data/trainingSample/img_8.jpg",0),cv2.imread("Data/trainingSample/img_3.jpg",0))
    
    return


def concatenateAndResize(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    #create empty matrix
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    
    #combine 2 images
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    resized_image = cv2.resize(vis, (28, 28)) 
    
    #plt.imshow(resized_image, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    
    return resized_image


if __name__ == "__main__":
    main()