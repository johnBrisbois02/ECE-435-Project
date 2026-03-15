import numpy as np
import math
import scipy

#Returns Gaussian Kernal of set size
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
    gaus_kernel = gaus_kernel * 1/sum
    sum = np.sum(gaus_kernel)


  return gaus_kernel 


def LoGKernal(wsize,sigma):

  kernel = GaussianKernal(wsize,sigma)

  offset = (wsize - 1)/2

  coef_factor = 2*math.pi
  
  sum = 0

  for i in range(wsize):
    for j in range(wsize):
      x = i - offset
      y = j - offset
      # gaussian kernel has coef of 1/ 2*pi*sigma^2
      # so multiplying by 2*pi*(x^2 + y^2 - 2sigma^2) will convert to LoG
      log_coef = ((x**2) + (y**2) - 2*(sigma**2)) / (sigma**2)
      kernel[i][j] = kernel[i][j] * coef_factor * log_coef

      sum = kernel[i][j] + sum
  #LoG kernal should be sum zero
  if sum > 0.05 or sum < -0.05:
    mean = sum/(wsize**2)
    kernel = kernel - mean
    sum = np.sum(kernel)

  return kernel



#convoles an image with a given kernal
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




#Convolutes image with a given kernal reutrns magnitude and angle, using sobel or Prewitt is usually considered the image gradient
def ImgGrad(img, mode = 'Sobel',):
  
  x_kernal = np.zeros((3,3))
  y_kernal = np.zeros((3,3))

  if mode == 'Prewitt':
    x_kernal = np.matrix([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    y_kernal = np.matrix([[1,1,1],[0,0,0],[-1,-1,-1]])
  elif mode == 'Sobel':
    x_kernal = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_kernal = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]])
  elif mode == 'Gravity':
    x_kernal = np.matrix([[-1*(math.sqrt(2)/4), 0, (math.sqrt(2)/4)], [-1, 0, 1], [-1*(math.sqrt(2)/4), 0, (math.sqrt(2)/4)]])
    y_kernal = np.matrix([[(math.sqrt(2)/4),1,(math.sqrt(2)/4)],[0,0,0],[-1*(math.sqrt(2)/4),-1,1*(math.sqrt(2)/4)]])

  

  else: #uses sobel for non implemented modes
    x_kernal = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_kernal = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]])

  grad_x = ImgConvolve(img,x_kernal)
  grad_y = ImgConvolve(img,y_kernal)


  GradImg = np.zeros(np.shape(img))
  ImgAng = np.zeros(np.shape(img))

  for i in range(np.shape(grad_x)[0]):
    for j in range(np.shape(grad_y)[1]):

      GradImg[i,j] = math.sqrt((grad_x[i,j]**2)+(grad_y[i,j]**2))
      ImgAng[i,j] = math.atan2(grad_y[i,j],grad_x[i,j])

  return GradImg, ImgAng
  
#Helps NonMaxSuppress function by giving correct directions
def SupressHelper(ang,y,x):
  #y is row, x is col
  #b is closest point a is second closest

  if ang < (math.pi/4):
    alpha = math.tan(ang)
    b1 = (y,x+1)
    a1 = (y-1,x+1)
    b2 = (y,x-1)
    a2 = (y+1,x-1)
    
  elif ang < (math.pi/2):
    alpha = 1/math.tan(ang)
    b1 = (y-1,x)
    a1 = (y-1,x+1)
    b2 = (y+1,x)
    a2 = (y+1,x-1)

  elif ang < (math.pi*(3/4)):
    alpha = -1*(1/math.tan(ang))
    b1 = (y-1,x)
    a1 = (y-1,x-1)
    b2 = (y+1,x)
    a2 = (y+1,x+1)

  else:
    alpha = -1*math.tan(ang)
    b1 = (y,x-1)
    a1 = (y-1,x-1)
    b2 = (y,x+1)
    a2 = (y+1,x+1)

  return alpha,a1,a2,b1,b2

#suppresses points where the direction of gradient isn't maximum
def NonMaxSuppress(img_mag,img_ang,thresh):

  img_sup = np.zeros(np.shape(img_mag))

  img_size = np.shape(img_mag)
  
  thresh_val = np.max(img_mag)*thresh
  

  val_1 = 0
  val_2 = 0
  alpha = 0
  a1 = 0
  a2 = 0
  b1 = 0
  b2 = 0

  
  for y in range(1,img_size[0]-1):
    for x in range(1,img_size[1]-1):

      if img_mag[y,x] > thresh_val:
        img_sup[y,x] = img_mag[y,x]
        continue
      

      cur_ang = abs(img_ang[y,x])
      alpha,a1,a2,b1,b2 = SupressHelper(cur_ang,y,x)
      val_1 = alpha*img_mag[b1] + (1-alpha)*img_mag[a1]
      val_2 = alpha*img_mag[b2] + (1-alpha)*img_mag[a2]

      if img_mag[y,x] > max(val_1,val_2):
        img_sup[y,x] = img_mag[y,x]

  return img_sup


def UpdateLevels(img_levels,follow_points,found):

  if found:
    for i in range(np.shape(img_levels)[0]):
      for j in range(np.shape(img_levels)[1]):
        if follow_points[i,j] == 0:
          continue
        if img_levels[i,j] == 1:
          img_levels[i,j] += 1 
          
  else:
    for i in range(np.shape(img_levels)[0]):
      for j in range(np.shape(img_levels)[1]):
        if follow_points[i,j] == 0:
          continue
        if img_levels[i,j] == 1:
          img_levels[i,j] -= 1 
          

  return

def FollowPath(img_levels,follow_points,cur_point,found):
  cur_i = cur_point[0]
  cur_j = cur_point[1]
  for i in range(-1,2):
    if cur_i+i<0 or cur_i+i>=np.shape(img_levels)[0]:
      continue
    for j in range(-1,2):
      if cur_j+j<0 or cur_j+j>=np.shape(img_levels)[1]:
        continue

      if img_levels[cur_i+i,cur_j+j] > 0 and follow_points[cur_i+i,cur_j+j] == 0:
        follow_points[cur_i+i,cur_j+j] = 1
        if img_levels[cur_i+i,cur_j+j] > 1:
          found = True
        found = FollowPath(img_levels,follow_points,(cur_i+i,cur_j+j),found)


  return found

# performs hysteresis
def Hysteresis(img,high_thresh,low_thresh):

  img_size = np.shape(img)

  img_levels = np.zeros(img_size)

  filt_img = np.zeros(img_size)

  high_val = high_thresh*np.max(img)
  low_val = low_thresh*np.max(img)

  for i in range(img_size[0]):
    for j in range(img_size[1]):
      if img[i,j]>high_val:
        #filt_img[i,j] = img[i,j] does this later anyway no need to do it twice
        img_levels[i,j] = 2
      elif img[i,j]>low_val:
        img_levels[i,j] = 1

  follow_points = np.zeros((img_size[0]+1,img_size[1]+1))

  for i in range(img_size[0]):
    for j in range(img_size[1]):
      if img_levels[i,j] != 1:
        continue
      for k in range(-1,2):
        if i+k<0 or i+k>=img_size[0]:
          continue
        for kk in range(-1,2):
          if j+kk<0 or j+kk>=img_size[1]:
            continue
          if img_levels[i+k,j+kk] == 1:
            follow_points[i,j] = 1
            found = FollowPath(img_levels,follow_points,(i+k,j+kk),False)
            UpdateLevels(img_levels,follow_points,found)
            follow_points[:,:] = 0

  for i in range(img_size[0]):
    for j in range(img_size[1]):
      if img_levels[i,j] >= 1:
        filt_img[i,j] = img[i,j]


  return filt_img

#Not working properly
def ImprovedCanny(img,gaus_size = 11, sigma = 1,mode = 'Gravity', max_thrsh = 0.8,k_coef = 1.4):

  gaus_kern = GaussianKernal(gaus_size, sigma)

  gaus_filt = ImgConvolve(img,gaus_kern)

  grad_img, grad_ang = ImgGrad(gaus_filt,mode)

  

  avg = 0

  img_shape = np.shape(img)

  

  for i in range(img_shape[0]):
    for j in range(img_shape[1]):
      avg += grad_img[i,j]

  avg = avg/(img_shape[0]*img_shape[1])

  imprv_sigma = 0

  for i in range(img_shape[0]):
    for j in range(img_shape[1]):
      imprv_sigma += (abs(grad_img[i,j] - avg))**2

  imprv_sigma = imprv_sigma / (img_shape[0]*img_shape[1])
  imprv_sigma = math.sqrt(imprv_sigma)
  high_thresh = avg + (k_coef*imprv_sigma)
  low_thresh = high_thresh/2

  max_supr_img = NonMaxSuppress(grad_img,grad_ang,high_thresh)

  filt_img = Hysteresis(max_supr_img,high_thresh,low_thresh)

  return filt_img


def ImgSharpen(img,sigma,kern_size):

  kernal = LoGKernal(kern_size, sigma)

  sharp_img = ImgConvolve(img,kernal)

  return sharp_img

#Canny edge detector algorithm
def CannyEdge(img,gaus_size = 11, sigma = 1,filt_mode = 'Gaussian', grad_mode = 'Sobel', max_thrsh = 0.8, hys_h_thrsh=0.7, hys_l_thrsh=0.2):

  

  if filt_mode == 'LoG':
    LoG_kernal = LoGKernal(gaus_size,sigma)
    first_filt = ImgConvolve(img,LoG_kernal)

  elif filt_mode == GaussianKernal:
    gaus_kern = GaussianKernal(gaus_size, sigma)
    first_filt = ImgConvolve(img,gaus_kern)

  else: #Gaussian Blur for unknown mode
    gaus_kern = GaussianKernal(gaus_size, sigma)
    first_filt = ImgConvolve(img,gaus_kern)
  

  grad_img, grad_ang = ImgGrad(first_filt,grad_mode)

  max_supr_img = NonMaxSuppress(grad_img,grad_ang,max_thrsh)

  filt_img = Hysteresis(max_supr_img,hys_h_thrsh,hys_l_thrsh)

  return filt_img


def Snake(img,x,y, alpha = 1, gamma = 1, beta = 1, GradT = 1, window_size = 11, k = 20):

  img_size = np.shape(img)

  #window size needs to be odd
  if window_size%2 == 0:
    window_size += 1

  eps = 1e-8

  global_norm = GradT * (np.max(img) - np.min(img))
  pos_buf = math.floor(window_size/2)
  padded_img = np.pad(img,pos_buf)

  x = x + pos_buf
  y = y + pos_buf
  x_prev = x[-1]
  y_prev = y[-1]

  numSeeds = len(x)

  Econt = np.zeros((window_size,window_size))
  Ecurv = np.zeros((window_size,window_size))


  for i in range(k):
    for j in range(numSeeds):
      nx_sd = j+1

      if nx_sd >= numSeeds:
        nx_sd = 0

      Econt = Econt*0
      Ecurv = Ecurv*0
      #col,row indexing for some reason
      Egrad = padded_img[(y[j]-pos_buf):(y[j]+pos_buf+1),(x[j]-pos_buf):(x[j]+pos_buf+1)]
      for w in range(-pos_buf,pos_buf+1):

        cur_x = x[j]+w
        for h in range(-pos_buf,pos_buf+1):
          cur_y = y[j]+h
          Econt[w+pos_buf,h+pos_buf] = (x[nx_sd]-cur_x)**2 +(cur_x- x_prev)**2 + (y[nx_sd]-cur_y)**2 + (cur_y - y_prev)**2
          Ecurv[w+pos_buf,h+pos_buf] = (x[nx_sd] - 2*(cur_x) + x_prev)**2 + (y[nx_sd]-2*(cur_y) + y_prev)**2

      #normalize

      cont_range = max((np.max(Econt)-np.min(Econt)),eps)
      Econt = Econt - np.min(Econt)
      Econt = Econt * (alpha/(2*cont_range))
      curv_range = max((np.max(Ecurv)-np.min(Ecurv)),eps)
      Ecurv = Ecurv - np.min(Ecurv)
      Ecurv = Ecurv * (beta/(2*curv_range))

      grad_norm = max((np.max(Egrad)-np.min(Egrad)),global_norm)
      Egrad = Egrad - np.min(Egrad)
      Egrad = Egrad *(gamma/grad_norm)
      Etot = Ecurv + Econt - Egrad

      min_row, min_col = np.unravel_index(np.argmin(Etot), Etot.shape)

      x_prev = x[j]
      y_prev = y[j]
      x[j] = x[j] + min_row - pos_buf
      y[j] = y[j] + min_col - pos_buf
      

  img_fin = np.zeros(img_size)

  for kk in range(numSeeds):
    img_fin[y[kk],x[kk]] = 255


  return img_fin