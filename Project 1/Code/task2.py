###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners
from matplotlib import pyplot as plt


def world_points():
    # world points
    wp_points=np.array([[40,0,40],[40,0,30],[40,0,20],[40,0,10],
    [30,0,40], [30,0,30],[30,0,20],[30,0,10],
    [20,0,40], [20,0,30],[20,0,20],[20,0,10],
    [10,0,40], [10,0,30],[10,0,20],[10,0,10],
    [0,0,40], [0,0,30],[0,0,20],[0,0,10],
    [0,10,40], [0,10,30],[0,10,20],[0,10,10],
    [0,20,40], [0,20,30],[0,20,20],[0,20,10],
    [0,30,40], [0,30,30],[0,30,20],[0,30,10],
    [0,40,40],[0,40,30],[0,40,20],[0,40,10]])
    
    return wp_points


def image_point(gray,chessboardSize,criteria,image,wp):
    #image points
    r, corners= findChessboardCorners(gray,chessboardSize,None)


    if r:
        
        
        corner_1 = cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corner_1 = corner_1.reshape(-1,2)
        drawChessboardCorners(gray,chessboardSize, corner_1, r)
        #print(corner_1.shape)
        #plt.imshow(image)



        M=np.zeros((2*36, 12), dtype=np.float64)


        for i in range(36):
            X, Y, Z = wp[i] 
            u, v = corner_1[i] 
            r_1 = np.array([ X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
            r_2 = np.array([ 0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
            M[2*i] = r_1
            M[(2*i) + 1] = r_2
        
    return M


def SVD(M):
    u,s, vh = np.linalg.svd(M)

    h_1 = vh[np.argmin(s)]
    h_1 = h_1.reshape(3, 4)
    
    return h_1


def intrinParameter(h):
    
    
    hmat=h
    M1=hmat[0:1,0:3]
    M2=hmat[1:2,0:3]
    M3=hmat[-1:,0:3]
    M4=hmat[:3,-1:]


    M10=M1.T
    M20=M2.T
    M30=M3.T
    M40=M4.T

 
    o_x=np.dot(M10.T,M30).item()
    o_y=np.dot(M20.T,M30).item()


    f_x=np.sqrt(np.dot(M10.T,M10)-o_x*o_x).item()
    f_y=np.sqrt(np.dot(M20.T,M20)-o_y*o_y).item()
    
    return o_x,o_y,f_x,f_y
    
    
    
def calibrate(img):
    img=imread(img)
    grayimage=cvtColor(img, COLOR_BGR2GRAY)
    cia=(TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER,30,0.001)
    
    
    
    # world points
    wp=world_points()
    
    
    
    #image points
    chessboardSize=(4,9)
    image_points = []
    M=image_point(grayimage,chessboardSize,cia,img,wp)
    
    


    #svd 
    h_1 = SVD(M)
    v1=h_1[-1,0]
    v2=h_1[-1,1]
    v3=h_1[-1,2]
    lamda=np.sqrt(1/(v1*v1+v2*v2+v3*v3))
    
    h_1=lamda*h_1
    

    


    #calculate intrinsic parameters
    o_x,o_y,f_x,f_y=intrinParameter(h_1)


   
    #return all parameters
    return  np.array([o_x,o_y,f_x,f_y]), True


if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)