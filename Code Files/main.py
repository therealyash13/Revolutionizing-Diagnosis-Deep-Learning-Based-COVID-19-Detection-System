from tkinter import *
import os
from PIL import Image, ImageTk
from tkinter import filedialog
from pymsgbox import *
import pandas as pd
import csv
from tkinter import messagebox
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
GREEN = '\033[92m'
RED = '\033[91m'


def endprogram():
	print ("\nProgram terminated!")
	sys.exit()

def getImage():
    global df
    
    import_file_path = filedialog.askopenfilename()
    
  
    image = cv2.imread(import_file_path)
    cv2.imshow('Original image',image)
    image = cv2.resize(image, (256,256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    cv2.imshow('Resized',image)
    cv2.imshow('Gray image', gray)
    #import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    fnm=os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))
   
    
    from PIL import Image, ImageOps

    im = Image.open(import_file_path)
    im_invert = ImageOps.invert(im)
    im = Image.open(import_file_path).convert('RGB')
    im_invert = ImageOps.invert(im)
        
    """"-----------------------------------------------"""
   
    img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original image',img)
    cv2.imshow('Gray image', gray)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    cv2.imshow("Nosie Removal",dst)
    import numpy as np
    from skimage import feature, io
    from sklearn import preprocessing

    img = io.imread(fnm,  as_gray=True)

    S = preprocessing.MinMaxScaler((0,11)).fit_transform(img).astype(int)
    Grauwertmatrix = feature.greycomatrix(S, [1,2,3], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=12, symmetric=False, normed=True)

    ContrastStats = feature.greycoprops(Grauwertmatrix, 'contrast')
    CorrelationtStats = feature.greycoprops(Grauwertmatrix, 'correlation')
    HomogeneityStats = feature.greycoprops(Grauwertmatrix, 'homogeneity')
    #print(ContrastStats)
    ASMStats = feature.greycoprops(Grauwertmatrix, 'ASM')
    
    glcm=[np.mean(ContrastStats), np.mean(CorrelationtStats), np.mean(ASMStats), np.mean(HomogeneityStats)]
    print("Feature Point:"+str(glcm))
    #cv2.imshow(glcm, cmap='gray');
def getimage1():
    import_file_path = filedialog.askopenfilename()
    
  
    image = cv2.imread(import_file_path)
    cv2.imshow('Original image',image)
    image = cv2.resize(image, (256,256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    cv2.imshow('Resized',image)
    cv2.imshow('Gray image', gray)
    #import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    fnm=os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))
    cv2.imwrite(fnm, image)
   
    
    from PIL import Image, ImageOps

    im = Image.open(import_file_path)
    im_invert = ImageOps.invert(im)
    im = Image.open(import_file_path).convert('RGB')
    im_invert = ImageOps.invert(im)
        
    """"-----------------------------------------------"""
   
    img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original image',img)
    cv2.imshow('Gray image', gray)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    cv2.imshow("Nosie Removal",dst)
    import numpy as np
    from skimage import feature, io
    from sklearn import preprocessing

    img = io.imread(fnm,  as_gray=True)

    S = preprocessing.MinMaxScaler((0,11)).fit_transform(img).astype(int)
    Grauwertmatrix = feature.greycomatrix(S, [1,2,3], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=12, symmetric=False, normed=True)

    ContrastStats = feature.greycoprops(Grauwertmatrix, 'contrast')
    CorrelationtStats = feature.greycoprops(Grauwertmatrix, 'correlation')
    HomogeneityStats = feature.greycoprops(Grauwertmatrix, 'homogeneity')
    #print(ContrastStats)
    ASMStats = feature.greycoprops(Grauwertmatrix, 'ASM')
    
    glcm=[np.mean(ContrastStats), np.mean(CorrelationtStats), np.mean(ASMStats), np.mean(HomogeneityStats)]
    print("Feature Point:"+str(glcm))
    import keras
    from keras.layers import Input
    from keras.models import load_model
    from keras.preprocessing import image as im
    model = load_model('model_adv.h5')
    
    img = im.load_img(fnm, target_size = (224, 224))
    img = im.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    p = model.predict_classes(img)
    print(p)
    print('Radiologist: COVID-19 -ve')
    if p==0:
            print('CNN Model: COVID-19 +ve')
    else:
            print("CNN Model: COVID-19 -ve")
        
    

        
    
    
    


# Designing popup for user not found


def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 300
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    #main_screen.geometry("300x250")
    main_screen.title("COVID Classification Pre-Processing")
   
   
    Label( text="COVID Classification Pre-Processing", bg="turquoise", width="300", height="5", font=("Calibri", 16)).pack()
    
    Button(main_screen,text='''Select Train Input Image''',font =('Verdana', 15), height="2", width="30", command = getImage).pack()
    Label(text="").pack()
    Button(main_screen,text='''Select Test Input Image''',font =('Verdana', 15), height="2", width="30", command = getimage1).pack()
    Label(text="").pack()
    
   
    main_screen.mainloop()


main_account_screen()
