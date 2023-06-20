from tkinter import *
from PIL import Image
import numpy as np
from PIL import Image as im
from tkinter import filedialog as tk
import tkinter as tk
import numpy as np
import copy
import math
import cv2
import random
from matplotlib import pyplot as plt
import gradio as gr


#Used to print Matrix
def printMatrix(matrix):
    print("\n-------------------------------------------------------\n")
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))
    print("\n-------------------------------------------------------\n")

#Used to shift image to range of -128 to 127 if needed
def shiftImage(imatrix):
    dimx = len(imatrix)
    dimy = len(imatrix[0])
    for i in range(dimx):
        for j in range(dimy):
            imatrix[i][j]-=128
    return imatrix

#Used to make a grid of non-overlapping windows determined by windowSize, returns a list of all submatrices of the grid
def makeWindows(imatrix,windowSize):
   
    dimx = len(imatrix)
    dimy = len(imatrix[0])
    windows=np.empty(shape=(dimx//windowSize,dimy//windowSize),dtype=object)
    cx=0
    cy=0
    tempMatrix=np.array(imatrix)
    for i in range(0,dimx,windowSize):
        for j in range(0,dimy,windowSize):
            windows[cx][cy]=tempMatrix[i:i+windowSize,j:j+windowSize]
            cy=(cy+1)%(dimy//windowSize)
        cx=(cx+1)%(dimx//windowSize)       
    return windows

#Preforms DCT function on matrix
def DCT(imatrix):
    tempMatrix = np.float32(imatrix) 
    dct = cv2.dct(tempMatrix, cv2.DCT_ROWS)
    return dct

#Function to apply DCT and quantize to all individual windows
def Apply_DCT(imatrix):
    imgWindows=makeWindows(imatrix,8)
    for windowRow in imgWindows:
        for window in windowRow:
            window=DCT(window)
            window=quantizeMatrix(window,quant_table) #Quantize window using quatize table
    return MergeMatrix(imgWindows,8)

#Takes in quantize table and divided each elemnt of matrix with corressponding quantize tabble element
def quantizeMatrix(imatrix,table):
    dimx = len(table)
    dimy= len(table[0])
    quantized=np.array(imatrix)
    for i in range(dimx):
        for j in range(dimy):
            quantized[i][j]=round(imatrix[i][j]/table[i][j])
    return quantized

#Block Trunication coding
def BTC(matrix,blockSize):
    global_bytes=matrix.nbytes
    
    blocks = makeWindows(matrix,blockSize) #Make window of matrix (usually 4x4)
    decoder=np.empty(shape=(len(blocks),len(blocks[0])),dtype=object) #Decoder list to keep decoding information
    count1=0
    count0=0
    x=0
    y=0
    for rowblock in blocks:
        for block in rowblock:
            mean = np.mean(block)
            std = np.std(block)
            for i in range(blockSize):
                for j in range(blockSize):
                    if(block[i][j]>=mean):   #For each element in window, if less than mean set to 0 else set to 1
                        block[i][j]=1
                        count1+=1
                    else:
                        block[i][j]=0
                        count0+=1
            if(count0==0):
                a=mean
            else:
                a=round(mean-(std*math.sqrt(count1/count0))) #Decoding information for 0s
            if(a<0):
                a*=-1
            if(a>255):
                a-=255
            if(count1==0):
                b=mean
            else:
                b=round(mean+(std*math.sqrt(count0/count1))) #Decoding information for 1s
            if(b<0):
                b*=-1
            if(b>255):
                b-=255

            count1=0
            count0=0
            decoder[x][y]=[a,b]
            
            y+=1

        x+=1
        y=0
    binaryMatrix = MergeMatrix(blocks,blockSize) #Merge windows back into one matrix
    #Converting int datatype (1 byte) to bool datatype(1 byte)
    binMat = np.empty(shape=(len(binaryMatrix),len(binaryMatrix[0])),dtype=np.bool_)
    for i in range(len(binaryMatrix)):
        for j in range(len(binaryMatrix[0])):
            if binaryMatrix[i][j]==1:
                binMat[i][j]=True
            else:
                binMat[i][j]=False
    #Covnerting bool datatype (1byte) to 1 bit as one bit is suffiecient 
    binMat = np.packbits(binMat,axis=None)
    

    global_bytes1=binMat.nbytes
    
    #Calculating compression ratio
    compr = matrix.nbytes//binMat.nbytes
    return binMat,decoder,compr,global_bytes,global_bytes1

#Use deocder information for every window to convert binary matrix back to grey level
def DecodeBTC(matrix,blockSize,decoder):
    #Converting 1 bit bool to 1 byte bool
    matrix = np.unpackbits(matrix, count=len(decoder)**2 * blockSize**2).reshape(blockSize*len(decoder),blockSize*len(decoder)).view(bool)
    blocks = makeWindows(matrix,blockSize)
    #Temp blocks to store resulted decoded matrix
    tempMat = np.empty(shape = (len(matrix),len(matrix[0])),dtype=object)
    tempBlocks = makeWindows(tempMat,blockSize)
    x=0
    y=0
    for blockRows in blocks:
        y=0
        for block in blockRows:
            for i in range(blockSize):
                for j in range(blockSize):
                    if(block[i][j]==False):
                        mean=decoder[x][y][0]
                        tempBlocks[x][y][i][j]= mean
                    else:
                        std=decoder[x][y][1]
                        tempBlocks[x][y][i][j]=std
            y+=1 
        x+=1
        y=0
    mMatrix = np.uint8(MergeMatrix(tempBlocks,blockSize))
    return mMatrix

#Returns a single matrix by merging all windows                 
def MergeMatrix(blocks,blockSize):
    mRow=[None for x in range(len(blocks[0]))]
    x=0
    y=0
    for rowblock in blocks:
        for block in rowblock:
            if(y==0):
                mRow[x]=np.array(block)
            else:
                mRow[x]=np.concatenate((mRow[x],block),axis=1)
            y+=1
        if(x==0):
            mMatrix=mRow[x]
        else:
            mMatrix=np.concatenate((mMatrix,mRow[x]),axis=0)    
        x+=1
        y=0
    return mMatrix

#Quantize table
quant_table = np.array(
[[16, 11, 10, 16, 24, 40, 51, 61],
[12, 12, 14, 19, 26, 58, 60, 55],
[14, 13, 16, 24, 40, 57, 69, 56],
[14, 17, 22, 29, 51, 87, 80, 62],
[18, 22, 37, 56, 68, 109, 103, 77],
[24, 35, 55, 64, 81, 104, 113, 92],
[49, 64, 78, 87, 103, 121, 120, 101],
[72, 92, 95, 98, 112, 100, 103, 99]])


# imatrix=cv2.imread("cameraman.tif",0)
# imatrix = Apply_DCT(imatrix)
# encodedMatrix,decoder,compRatio = BTC(imatrix,4)
# print("Compression Ratio is -> ", str(compRatio), " : 1")
# decodedMatrix = DecodeBTC(encodedMatrix,4,decoder)
# cv2.imshow("Compressed Image",decodedMatrix)

def compress():
    f_types=[('Jpg files','*.jpg'),('PNG files','*.png'),('Tif Files','*.tif')]
    filename=tk.filedialog.askopenfilename(filetypes=f_types)
    imatrix=cv2.imread(filename,0)
    imatrix = Apply_DCT(imatrix)
    encodedMatrix,decoder,compRatio,g_bytes_org,g_bytes_compressed = BTC(imatrix,4)

   
    decodedMatrix = DecodeBTC(encodedMatrix,4,decoder)
    cv2.imshow("Compressed Image",decodedMatrix)
    s=tk.Tk()
    s.geometry('600x500')
    s.title("Properties")
    label1=Label(s, text="Properties",fg='black',font=('Arial',20))
    label1.place(x=65,y=20)
    label2=Label(s, text="Size of Orignal image in bytes:",fg='black',font=('Arial',14))
    label2.place(x=65,y=100)
    label3=Label(s, text=str(g_bytes_org),fg='black',font=('Arial',14))
    label3.place(x=350,y=100)
    label4=Label(s, text="Size of Compressed image in bytes:",fg='black',font=('Arial',14))
    label4.place(x=65,y=200)
    label5=Label(s, text=str(g_bytes_compressed),fg='black',font=('Arial',14))
    label5.place(x=400,y=200)
    label6=Label(s, text="Compressed Image Ratio:",fg='black',font=('Arial',14))
    label6.place(x=65,y=300)
    label7=Label(s, text=str(compRatio),fg='black',font=('Arial',14))
    label7.place(x=350,y=300)
    label8=Label(s, text=" :1",fg='black',font=('Arial',14))
    label8.place(x=365,y=300)

window=tk.Tk()
window.geometry('400x400')
window.title("Image Compression")

frame=tk.Frame(window,borderwidth=6, bg="grey",relief=SUNKEN)
frame.pack(side=LEFT, anchor="nw")

label1=Label(window, text="Image Compression",fg='black',font=('Arial',25))
label1.place(x=65,y=40)
button2=tk.Button(window,text='UPLOAD IMAGE', fg='black',font=('Arial',14),command=compress)
button2.place(x=120,y=165)


window.mainloop()
#cv2.waitKey(0)
# cv2.destroyAllWindows()




