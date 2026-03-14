
from functions import  CannyEdge, ImprovedCanny, Snake
#GaussianKernal, ImgConvolve, ImgGrad, NonMaxSuppress, Hysteresis now internal functions
from PIL import Image
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def main():

    img_stack = tifffile.imread('Project_Data/2_image.tiff')

    #img = Image.fromarray(img_stack[30,:,:])
    #img.show()

    cur_img = img_stack[30,:,:]

    #def CannyEdge(img,gaus_size = 11, sigma = 1,mode = 'Sobel', max_thrsh = 0.8, hys_h_thrsh=0.7, hys_l_thrsh=0.2)
    #function description
    canny_img = CannyEdge(cur_img,gaus_size = 11, sigma = 1,mode= 'Sobel', 
                          max_thrsh = 0.7, hys_h_thrsh=0.7, hys_l_thrsh=0.3)

    #imprv_canny = ImprovedCanny(cur_img,gaus_size = 11, sigma = 1,mode= 'Gravity', 
                          #max_thrsh = 0.8, k_coef=2)


    #Snake(img,x,y, alpha = 1, gamma = 1, beta = 1, GradT = 1, window_size = 11, k = 20):

    x = np.array([83,120,127,120,90,55,45,50])
    y = np.array([85,100,140,180,185,167,135,100])

    

    #snake_img = Snake(canny_img,x,y, alpha = 0.3, gamma = 2, beta = 0.5, GradT = 0.5, window_size = 11, k = 20)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(cur_img,cmap='grey')
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1,2,2)
    plt.imshow(canny_img,cmap='grey')
    plt.axis('off')
    plt.title("Canny Edge")
    
    plt.show()

    


    return


main()
