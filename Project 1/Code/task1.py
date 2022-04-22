###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
#It is ok to add other functions if you need
###############

import numpy as np
import cv2

def Rmatrix(a,b,c):
    
    Rz=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
    Rx=[[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]]
    RZ=[[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]]
    x=np.matrix(Rz)
    y=np.matrix(Rx)
    z=np.matrix(RZ)
    ans2=y*x
    ans3=z*ans2
    
    
    return ans3


def Rinvmatrix(a,b,c):
    
    
    RZr=[[np.cos(-c),-np.sin(-c),0],[np.sin(-c),np.cos(-c),0],[0,0,1]]
    Rxr=[[1,0,0],[0,np.cos(-b),-np.sin(-b)],[0,np.sin(-b),np.cos(-b)]]
    Rzr=[[np.cos(-a),-np.sin(-a),0],[np.sin(-a),np.cos(-a),0],[0,0,1]]

    m=np.matrix(RZr)
    n=np.matrix(Rxr)
    o=np.matrix(Rzr)

    res1=n*m

    resfin=o*res1
    
    return resfin
    
def findRotMat(alpha, beta, gamma):
    
    a=(alpha/180)*np.pi
    b=(beta/180)*np.pi
    c=(gamma/180)*np.pi
    

    rmatrix=Rmatrix(a,b,c)
    rinvmatrix=Rinvmatrix(a,b,c)
    
 

    return rmatrix, rinvmatrix


if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)