import os

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

trainingSetPath = './Data/trainingSet'
sentenceTypes = []
sentenceTypes.append("Multiply {number1} by {number2}")
sentenceTypes.append("Square {number}")

def main():
    labelDirs = dir_list = next(os.walk(trainingSetPath))[1]
    print(labelDirs)
    filenames = []
    for dir in labelDirs:
        path = trainingSetPath+'/'+str(dir)
        filenames.append(os.listdir())



    return


if __name__ == "__main__":
    main()