# import
from PIL import ImageDraw, Image
from random import randrange
from os import sep
import pandas as pd
import numpy as np

# steps:
# 1. build a function to create as many random images as passed as parameter
# 2. build a function to create training & test data separation

def fn_make_img(count=100, path="data"):

    # check if master file already exist if yes then delete and recreate

    # creating file header
    with open(path + sep + "tracking_master.csv", mode="w+") as f:
        f.writelines("Image_Name,Flag\n")

    # save the call and the corelating image name in a csv
    # this is our lookup file, also this file will help to create training vs test data in future
    with open(path + sep + "tracking_master.csv", mode="a") as f:
        # for decision purpose run a for loop and for each step
        # take a random call of what class of image to be generated
        for i in range(0,count):
            # print(randrange(0,4))
            # 0 - Blank
            # 1 - Green
            # 2 - Yellow
            # 3 - Red
            rand_call = randrange(0, 4)

            # if rand_call == 0:
            #     rand_call = "Blank"
            # elif rand_call == 1:
            #     rand_call = "Green"
            # elif rand_call == 2:
            #     rand_call = "Yellow"
            # elif rand_call == 3:
            #     rand_call = "Red"

            # save the call and the corelating image name in a csv
            # this is our lookup file, also this file will help to create training vs test data in future
            f.writelines(str(i)+".png,"+str(rand_call)+"\n")

            # generate the image

            new_img = Image.new('RGB',(32,32),color='white')

            # create rectangular border
            draw = ImageDraw.Draw(new_img)
            draw.rectangle(((0, 0), (31, 31)), fill="white",outline="black")


            # create circle inside
            # color circle based on the flag of what kind of image it is
            if rand_call == 0:
                draw.ellipse([4,4,28,28],outline="Black")
            elif rand_call == 1:
                draw.ellipse([4,4,28,28],outline="Black",fill="green")
            elif rand_call == 2:
                draw.ellipse([4,4,28,28],outline="Black",fill="yellow")
            elif rand_call == 3:
                draw.ellipse([4,4,28,28],outline="Black",fill="red")


            del draw
            # save image
            new_img.save(path + sep + str(i)+".png")

    return

def split(df, headSize) :
    hd = df.head(headSize)
    tl = df.tail(len(df)-headSize)
    return hd, tl

def load_data(img_path="data",path="data"+sep+"tracking_master.csv", split_ratio=0.8):

    # first open the file that needs to be slitted
    data = pd.read_csv(path,sep=',')

    # print(data.head())

    # split data
    print(data.shape[0])

    training_data_tracker, test_data_tracker = split(data, int(data.shape[0]*split_ratio))
    print(training_data_tracker.shape)
    print(test_data_tracker.shape)

    # once the split is done
    # we need to loop through the training and test set,
    # and for each image we need to open the image
    # and conver the data into a multi dimentional array
    # where the array's structure should be width, height, channel information
    # so the shape should be [32,32,3]

    # + there is another dimention to consider, that is the number of images
    # so we will create a placeholder first for so many files and then put the data in the
    #  placeholder as and when generated

    # in order to build the placeholder dynamic we need to take the height & width info
    #  from one of the images
    op_img = Image.open(img_path + sep + data.ix[0][0])
    (width, height) = op_img.size

    # also we can try to make the channel information dynamic
    # monochrome-grayscale
    if len(np.array(op_img).shape) == 2:
        channel = 1
    # color images
    else:
        channel = np.array(op_img).shape[2]

    # lets take care of the target part separately first
    y_train = training_data_tracker["Flag"]
    y_test = test_data_tracker["Flag"]

    # now lets look into the feature part
    # create a default placeholder of <number of images>,<width>,<height>,<channel>
    # for training data
    x_train = np.zeros((training_data_tracker.shape[0], width, height, channel), dtype='uint8')

    for i in training_data_tracker.index:
        print(training_data_tracker.ix[i][0])
        # imgname = data.iloc[i,]['Image_Name'] # another way to get the image name
        op_img = Image.open(img_path+sep+training_data_tracker.ix[i][0])
        (width, height) = op_img.size
        op_img_map = list(op_img.getdata())
        op_img_map = np.array(op_img_map)
        x_train[i,:,:,:] = op_img_map.reshape(width, height, channel)


    # for test data
    x_test = np.zeros((test_data_tracker.shape[0], width, height, channel), dtype='uint8')

    test_data_tracker = test_data_tracker.reset_index(drop=True)
    for i in test_data_tracker.index:
        print(test_data_tracker.ix[i][0])
        # imgname = data.iloc[i,]['Image_Name'] # another way to get the image name
        op_img = Image.open(img_path+sep+test_data_tracker.ix[i][0])
        (width, height) = op_img.size
        op_img_map = list(op_img.getdata())
        op_img_map = np.array(op_img_map)
        x_test[i,:,:,:] = op_img_map.reshape(width, height, channel)

    return x_train,y_train,x_test,y_test

# ==================================================================
# main
# fn_make_img(count=10000)
load_data()