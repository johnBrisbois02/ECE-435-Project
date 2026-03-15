
from functions import  CannyEdge, ImprovedCanny, Snake, ImgSharpen, HistEqual
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
    

    #CannyEdge(img,gaus_size = 11, sigma = 1,filt_mode = 'Gaussian', grad_mode = 'Sobel', max_thrsh = 0.8, hys_h_thrsh=0.7, hys_l_thrsh=0.2):
    #function description
    canny_img = CannyEdge(cur_img,gaus_size = 11, sigma = 2, filt_mode='Gaussian' ,grad_mode= 'Sobel', 
                          max_thrsh = 0.7, hys_h_thrsh=0.6, hys_l_thrsh=0.3)

    #canny_img_prewitt = CannyEdge(cur_img,gaus_size = 11, sigma = 2 ,mode= 'Prewitt', 
    #                      max_thrsh = 0.7, hys_h_thrsh=0.6, hys_l_thrsh=0.3)

    #equal_img = HistEqual(cur_img)

    sharpened_img = ImgSharpen(cur_img,sigma = 2, kern_size=17)

    sharpened_img = sharpened_img - np.min(sharpened_img)
    sharpened_img = sharpened_img * 255/(np.max(sharpened_img))


    sharp_canny = CannyEdge(cur_img,gaus_size = 11, sigma = 0.6, filt_mode='LoG' ,grad_mode= 'Sobel', 
                          max_thrsh = 0.7, hys_h_thrsh=0.6, hys_l_thrsh=0.3)

    #imprv_canny = ImprovedCanny(cur_img,gaus_size = 11, sigma = 1,mode= 'Gravity', 
                          #max_thrsh = 0.8, k_coef=2)
    

    #x = np.array([83,120,127,120,90,55,45,50])
    #y = np.array([85,100,140,180,185,167,135,100])
    #snake_img = Snake(canny_img,x,y, alpha = 0.3, gamma = 2, beta = 0.5, GradT = 0.5, window_size = 11, k = 20)

    plt.figure(1)
    
    plt.imshow(canny_img,cmap='grey')
    plt.axis('off')
    plt.title("Canny Edge")
    
    plt.figure(2)
    plt.imshow(sharpened_img,cmap='grey')
    plt.axis('off')
    plt.title("LoG sharpened")

    plt.figure(3)
    plt.imshow(cur_img,cmap='grey')
    plt.axis('off')
    plt.title("Original")
    

    plt.figure(4)
    plt.imshow(sharp_canny,cmap='grey')
    plt.axis('off')
    plt.title("Sharpened Canny")
    plt.show()

    return


main()
