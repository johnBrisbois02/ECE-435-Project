
from functions import  CannyEdge
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
                          max_thrsh = 0.8, hys_h_thrsh=0.6, hys_l_thrsh=0.3)




    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(cur_img,cmap='grey')
    plt.axis('off')
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(canny_img,cmap='grey')
    plt.axis('off')
    plt.title("Canny Edge Detector")
    plt.show()

    


    return


main()
