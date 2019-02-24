
# coding: utf-8

# In[40]:


import numpy as np
import os
from PIL import Image
import cv2
import pandas as pd
import os
import glob
import random


# In[59]:


# function below draws the predict bounding box and the ground truth bounding box
# reference: https://github.com/datitran/raccoon_dataset/blob/master/draw%20boxes.ipynb
def draw_boxes(image_name, imagedir, model, predictLables, GroundTruthLabels):
    Predicted_value = predictLables[predictLables.ImageID == image_name]
    GroundTruth_value = GroundTruthLabels[GroundTruthLabels.filename == image_name]
    Predicted_value =  Predicted_value[Predicted_value.Model == model ]
    img = cv2.imread(imagedir+image_name)
    Width = pd.unique(Predicted_value.Width)[0]
    Height = pd.unique(Predicted_value.Height)[0]
    print(Height)
    #print(Width)
    #print(GroundTruth_value)
    #Draw Predicted BoundingBox
    for index, row in Predicted_value.iterrows():
        #<class_name> <confidence> <left> <top> <right> <bottom>
        img = cv2.rectangle(img, (round(row.left*row.Width), round(row.top*row.Height)), (round(row.right*row.Width), round(row.bottom*row.Height)), (255, 0, 0), 3)
        #print(round(row.left*row.Width), round(row.top*row.Height), round(row.right*row.Width), round(row.bottom*row.Height))
    for index, row in GroundTruth_value.iterrows():
        img = cv2.rectangle(img, (row.xmin,row.ymin),(row.xmax,row.ymax),(0, 255, 0), 3)
        
    return img

def draw_boxes_GT(image_name, imagedir, model, predictLables, GroundTruthLabels):
    Predicted_value = predictLables[predictLables.ImageID == image_name]
    GroundTruth_value = GroundTruthLabels[GroundTruthLabels.filename == image_name]
    Predicted_value =  Predicted_value[Predicted_value.Model == model ]
    img = cv2.imread(imagedir+image_name)
    Width = pd.unique(Predicted_value.Width)[0]
    Height = pd.unique(Predicted_value.Height)[0]
    print(Height)
    #print(Width)
    #print(GroundTruth_value)
    #Draw Predicted BoundingBox
    #for index, row in Predicted_value.iterrows():
    #    #<class_name> <confidence> <left> <top> <right> <bottom>
    #    img = cv2.rectangle(img, (round(row.left*row.Width), round(row.top*row.Height)), (round(row.right*row.Width),
    #                                                                                      round(row.bottom*row.Height)), (255, 0, 0), 3)
    #    #print(round(row.left*row.Width), round(row.top*row.Height), round(row.right*row.Width), round(row.bottom*row.Height))
    for index, row in GroundTruth_value.iterrows():
        img = cv2.rectangle(img, (row.xmin,row.ymin),(row.xmax,row.ymax),(0, 255, 0), 3)
        
    return img, GroundTruth_value

"""
 Draws text in image
"""
def draw_text_in_image(img, text ='car', pos = (235,137), color=128, line_width=8):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 2
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img#, (line_width + text_width)



# In[63]:
#####
#
#This function takes a path to the desired 
#folder and damples some bounding boxes for a default of 2 images
######
def run (path = "samples-1k/Driving_condition/night", n=2):  
    # Loading ground truth  and Predict labels from csv file
    condition = path.split('/')[2]
    for File in  glob.glob(path + '/*groundTruthlabels.csv'):
        GroundTruth =pd.read_csv(File)
    Predicted= pd.read_csv('InferenceData.csv')
    temp =Predicted[Predicted['Driving Condition']==condition]
    # Prulling image list and models type from the loaded file
    imageslist  =pd.unique(temp.ImageID)
    models = pd.unique(temp.Model)

    print("RED -> Predicted")
    print("GREEN -> Ground Truth")

    for i in imageslist[0:n]: #pulling on the first two images

        display(Image.fromarray(draw_boxes(i, path+'/images/',models[3] , temp, GroundTruth)))
        
        m, n= draw_boxes_GT(i, path+'/images/',models[3] , temp, GroundTruth)
        display(Image.fromarray(m))
        display(n)
        display(Image.fromarray(draw_text_in_image(m, "car",  (pd.unique(GroundTruth.xmin)[0],pd.unique(GroundTruth.ymin)[0]), (0,255,0), 0.8)))
    return(m)
                

# In[64]:




