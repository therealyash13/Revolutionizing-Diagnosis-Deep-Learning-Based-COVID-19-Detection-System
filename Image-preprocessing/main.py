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
    #imshow(glcm, cmap='gray');
        

    
    
    
    


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
    
    Button(main_screen,text='''Select Input Image''',font =('Verdana', 15), height="2", width="30", command = getImage).pack()
    Label(text="").pack()
    
   
    main_screen.mainloop()


main_account_screen()
