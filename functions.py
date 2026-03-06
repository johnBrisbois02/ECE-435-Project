import numpy as np
import math
import scipy

def GaussianKernal(wsize,sigma):

  if wsize % 2 == 0:
    print("Size input must be odd")
    return -1

  gaus_kernel = np.zeros((wsize,wsize))
  sum = 0

  offset = (wsize-1)/2
  two_sigma_sqrd = 2*(sigma**2)

  for i in range(wsize):
    for j in range(wsize):
      x_sqrd = (i-offset)**2
      y_sqrd = (j-offset)**2
      expon = math.exp(-1*((x_sqrd+y_sqrd)/two_sigma_sqrd))
      gaus_kernel[i][j] = (1/(math.pi * two_sigma_sqrd))*expon
      sum = sum + gaus_kernel[i][j]

  if sum > 1.1 or sum < 0.9:
    kernel = kernel * 1/sum
    sum = np.sum(kernel)


  out = gaus_kernel #Gaussian normalized


def ImgConvolve(img, kernel):
  

  max_val = np.max(img)
  min_val = np.min(img)

  if max_val > 255 or min_val < 0:

    im_range = max_val - min_val
    img = img - min_val
    img = img * 1/im_range
    img = img*255

  #img_conv = scipy.ndimage.convolve(img,kernel,mode='constant', cval = 0)
  img = img.astype(np.float64)

  img_conv = scipy.ndimage.convolve(img,kernel)

  np.clip(img_conv,0,255)

  return img_conv


def SobelGrad(img):
  
    sob_kernal_x = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sob_kernal_y = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]])

    sob_grad_x = ImgConvolve(img,sob_kernal_x)
    sob_grad_y = ImgConvolve(img,sob_kernal_y)


    SobelGradImg = np.zeros(np.shape(img))

    for i in range(np.shape(sob_grad_x)[0]):
        for j in range(np.shape(sob_grad_x)[1]):

            SobelGradImg[i][j] = math.sqrt((sob_grad_x[i,j]**2)+(sob_grad_y[i,j]**2))

    return SobelGradImg