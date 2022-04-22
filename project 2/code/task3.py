# -*- coding: utf-8 -*-
"""cvipfinaltask3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y-MUOAZTs8ga6uoqfd-DVIMGqRYFw_1w
"""



"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""
import cv2
#from google.colab.patches import cv2_imshow
from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """
    #cv2_imshow(img)
    #print(img.shape)
    

    filter = np.array([(1, 1, 1), (1, 1,1), (1, 1, 1)])
    filt_shape = filter.shape
    img_shape = img.shape
    #print(img_shape)
    imag = img/255
    m = img_shape[0]+filt_shape[0]-1
    n = img_shape[1]+filt_shape[1]-1
    #print(m)
    #print(n)
    output = np.zeros((m,n), dtype=np.uint8)
    for i in range(img_shape[0]):
      for j in range(img_shape[1]):
        output[i+1,j+1] = imag[i,j]
    k=3   
    con = (k-1)//2
    a= m-2
    b= n-2

    for k in range(2):
      for i in range(con, a-con):
        for j in range(con, b-con):
          k = img[i-con:i+con+1, j-con:j+con+1]
          result = filter*k
          output[i,j] = np.min(result)
     


    
    #cv2_imshow(output)
    # TO DO: implement your solution here
    
    return output

def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """
    #cv2_imshow(img)
    #print(img.shape)
    

    filter = np.array([[1, 1, 1], [1, 1,1], [1, 1, 1]])
    filt_shape = filter.shape
    img_shape = img.shape
    #print(img_shape)
    imag = img/255
    m = img_shape[0]+filt_shape[0]-1
    n = img_shape[1]+filt_shape[1]-1
    #print(m)
    #print(n)
    output = np.zeros((m,n), dtype=np.uint8)
    for i in range(img_shape[0]):
      for j in range(img_shape[1]):
        output[i+1,j+1] = imag[i,j]
    k=3   
    con = (k-1)//2
    a= m-2
    b= n-2

    for i in range(1, a-1):
      for j in range(1, b-1):
        k = img[i-1:i+1+1, j-1:j+1+1]
        result = k*filter
        output[i,j] = np.max(result)
    #cv2_imshow(output)
    # TO DO: implement your solution here
    #raise NotImplementedError
    return output

def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    open_img = morph_erode(img)
    # TO DO: implement your solution here
    
    return open_img

def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    close_img = morph_dilate(img)
    close_img = morph_erode(close_img)
    # TO DO: implement your solution here
    #raise NotImplementedError
    return close_img

def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """
    #cv2_imshow(img)
    #print(img.shape)
    img = morph_open(img)
    denoise_img = morph_close(img)
    #cv2_imshow(denoise_img)
    #print(denoise_img.shape)
    #denoise_img = cv2.resize(denoise_img,(500, 500))
    # TO DO: implement your solution here
    #raise NotImplementedError
    return denoise_img

def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """
    
    img1 = morph_erode(img)
    img1 = cv2.resize(img1,(500, 500))
    img = cv2.resize(img,(500, 500))
    bound_img = img - img1
    #cv2_imshow(bound_img)
    #TO DO: implement your solution here
    #raise NotImplementedError
    return bound_img

if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)



