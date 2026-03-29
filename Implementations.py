
from functions import  CannyEdge, TenengradMask, RegionGrow, ImprovedCanny, FocusDiffernceMask
#GaussianKernal, ImgConvolve, ImgGrad, NonMaxSuppress, Hysteresis now internal functions
from PIL import Image
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import cv2

def main():


    
    img_stack = tifffile.imread('Project_Data/2_image.tiff')

    back_masks,for_masks,cen_masks = FocusDiffernceMask(img_stack,window_size = 19, kernel_mode = 4, derivative = 'backward',thresh = 0.5)
    
    
    tifffile.imwrite("back_dif_2.tiff",back_masks)
    tifffile.imwrite("for_dif_2.tiff",for_masks)
    tifffile.imwrite("cen_dif_2.tiff",cen_masks)

    return

    '''
    #masks = np.zeros_like(img_stack)
    ten_lap_masks = np.zeros_like(img_stack)

    for i in range(np.shape(img_stack)[0]):
    
        cur_img = img_stack[i,:,:]
        
        print(i)
        
        #canny_img = CannyEdge(cur_img,gaus_size = 11, sigma = 1, filt_mode='Gaussian' ,grad_mode= 'Sobel', 
        #                    max_thrsh = 0.6, hys_h_thrsh=0.8, hys_l_thrsh=0.3)

        canny_img = cv2.Canny(cur_img,100,175)
        #alpha for LoG, beta for Edge density (canny), gamma for gradient
        #img_mask = RegionGrow(cur_img, canny_img, gaus_size = 11,sigma =1, focus_thresh = 0.4, sim_thresh = 4, seed_thresh = 0.8,
        #                      edge_stop_thresh = 0.4, alpha = 0.6, beta = 0.05, gamma = 0.7, window_size = 11)

        #alpha for tenengrad, beta for laplacian variance, gamma for edge density
        ten_lap_mask = TenengradMask(cur_img,canny_img,window_size = 11, alpha = 0.35, beta = 1, gamma = 0.1,
                                     seed_thresh = 0.8,grow_thresh = 0.7)

        #masks[i,:,:] = img_mask*255
        ten_lap_masks[i,:,:] = ten_lap_mask*255


    dif_masks = FocusDiffernceMask(img_stack,window_size = 11, kernel_mode = 4, derivative = 'backward',thresh = 0.5)
    tifffile.imwrite("lap_dif_2.tiff",dif_masks)
    #tifffile.imwrite("masks_2.tiff",masks)
    tifffile.imwrite("ten_lap_masks_2.tiff",ten_lap_masks)
    '''

    '''
    img_stack = tifffile.imread('Project_Data/3_image.tiff')
    
    masks = np.zeros_like(img_stack)

    ten_lap_masks = np.zeros_like(img_stack)

    for i in range(np.shape(img_stack)[0]):
    
        cur_img = img_stack[i,:,:]
        
        print(i)
        
        
        canny_img = CannyEdge(cur_img,gaus_size = 11, sigma = 1, filt_mode='Gaussian' ,grad_mode= 'Sobel', 
                            max_thrsh = 0.6, hys_h_thrsh=0.8, hys_l_thrsh=0.3)

        img_mask = RegionGrow(cur_img, canny_img, gaus_size = 11,sigma =1, focus_thresh = 0.4, sim_thresh = 4, seed_thresh = 0.8,
                              edge_stop_thresh = 0.4, alpha = 0.6, beta = 0.05, gamma = 0.7, window_size = 11)

        ten_lap_mask = TenengradMask(cur_img,canny_img,window_size = 11, alpha = 1, beta = 1, gamma = 0.2,
                                     seed_thresh = 0.8,grow_thresh = 0.3)

        
        masks[i,:,:] = img_mask*255
        ten_lap_masks[i,:,:] = ten_lap_mask*255

   
    tifffile.imwrite("ten_lap_masks_3.tiff",ten_lap_masks)
    tifffile.imwrite("masks_3.tiff",masks)

    
    

    img_stack = tifffile.imread('Project_Data/6_image.tiff')
    masks = np.zeros_like(img_stack)
    ten_lap_masks = np.zeros_like(img_stack)

    for i in range(np.shape(img_stack)[0]):
        cur_img = img_stack[i,:,:]
        
        print(i)
        
        canny_img = CannyEdge(cur_img,gaus_size = 11, sigma = 1, filt_mode='Gaussian' ,grad_mode= 'Sobel', 
                            max_thrsh = 0.6, hys_h_thrsh=0.8, hys_l_thrsh=0.3)

        img_mask = RegionGrow(cur_img, canny_img, gaus_size = 11,sigma =1, focus_thresh = 0.4, sim_thresh = 4, seed_thresh = 0.8,
                              edge_stop_thresh = 0.4, alpha = 0.6, beta = 0.05, gamma = 0.7, window_size = 11)
        
        ten_lap_mask = TenengradMask(cur_img,canny_img,window_size = 11, alpha = 1, beta = 1, gamma = 0.2,
                                     seed_thresh = 0.8,grow_thresh = 0.3)

        
        masks[i,:,:] = img_mask*255
        ten_lap_masks[i,:,:] = ten_lap_mask*255

   
    tifffile.imwrite("ten_lap_masks_6.tiff",ten_lap_masks)
    tifffile.imwrite("masks_6.tiff",masks)


    img_stack = tifffile.imread('Project_Data/7_image.tiff')
    masks = np.zeros_like(img_stack)
    ten_lap_masks = np.zeros_like(img_stack)

    for i in range(np.shape(img_stack)[0]):
        cur_img = img_stack[i,:,:]
        
        print(i)
        
        
        canny_img = CannyEdge(cur_img,gaus_size = 11, sigma = 1, filt_mode='Gaussian' ,grad_mode= 'Sobel', 
                            max_thrsh = 0.6, hys_h_thrsh=0.8, hys_l_thrsh=0.3)

        img_mask = RegionGrow(cur_img, canny_img, gaus_size = 11,sigma =1, focus_thresh = 0.4, sim_thresh = 4, seed_thresh = 0.8,
                              edge_stop_thresh = 0.4, alpha = 0.6, beta = 0.05, gamma = 0.7, window_size = 11)

        ten_lap_mask = TenengradMask(cur_img,canny_img,window_size = 11, alpha = 1, beta = 1, gamma = 0.2,
                                     seed_thresh = 0.8,grow_thresh = 0.3)

        
        masks[i,:,:] = img_mask*255
        ten_lap_masks[i,:,:] = ten_lap_mask*255

   
    tifffile.imwrite("ten_lap_masks_7.tiff",ten_lap_masks)
    tifffile.imwrite("masks_7.tiff",masks)


    '''
        
    '''
    plt.figure(1)
    
    plt.imshow(ten_lap_mask,cmap='grey')
    plt.axis('off')
    plt.title("Tenegrad and Laplacian Variance Mask")
    
    plt.figure(2)
    plt.imshow(img_mask,cmap='grey')
    plt.axis('off')
    plt.title("Image Mask")

    plt.figure(3)
    plt.imshow(cur_img,cmap='grey')
    plt.axis('off')
    plt.title("Original")
    
    plt.show()
    '''
    


    return

    

main()
