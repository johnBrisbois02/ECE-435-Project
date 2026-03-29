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
def ImgGrad(img, mode = 'Sobel', val_mode = 'default'):
  
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
      if val_mode == "Tenengrad":
        GradImg[i,j] = (grad_x[i,j]**2)+(grad_y[i,j]**2)
      else:
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



def EdgeDensity(canny_img,loc,window_size):
  img_shape = np.shape(canny_img)
  start_i = loc[0]-math.floor(window_size/2)
  start_j = loc[1]-math.floor(window_size/2)

  end_i = loc[0]+math.floor(window_size/2)+1
  end_j = loc[1]+math.floor(window_size/2)+1

  if end_i>=img_shape[0]:
    end_i = img_shape[0]
  if end_j>=img_shape[1]:
    end_j = img_shape[1]

  val = np.sum(canny_img[start_i:end_i,start_j:end_j]) / ((end_i-start_i)*(end_j - start_j))
  #val = np.sum(canny_img[start_i:end_i,start_j:end_j])
  return val

#start with tenegrad, returns pixel (location_val - avg)^2
def FocusValMean(img,loc,window_size):

  
  start_i = loc[0]-math.floor(window_size/2)
  start_j = loc[1]-math.floor(window_size/2)

  img_shape = np.shape(img)

  end_i = loc[0]+math.floor(window_size/2)+1
  end_j = loc[1]+math.floor(window_size/2)+1

  if end_i>=img_shape[0]:
    end_i = img_shape[0]
  if end_j>=img_shape[1]:
    end_j = img_shape[1]


  norm_val = (end_i-start_i) * (end_j-start_j)

  avg = np.sum(img[start_i:end_i,start_j:end_j])

  

  return (avg/norm_val)


def CheckNeighbour(focus_img,mask_img,focus_thresh,canny_img,sim_thresh,loc,window_size,edge_stop_thresh):

  loc_i = loc[0]
  loc_j = loc[1]
  changed = False

  for i in range(-1,2):
    for j in range(-1,2):
      if mask_img[loc_i+i,loc_j+j] == 1:
        continue
      #elif (canny_img[loc_i+i,loc_j+j] - canny_img[loc_i,loc_j]) > edge_stop_thresh:
      #  continue
      elif focus_img[loc_i+i,loc_j+j] < focus_thresh:
        #print("Focus Thresh:%f Current Val: %f",{focus_thresh,focus_img[loc_i+i,loc_j+j]})
        continue
      
      elif abs(focus_img[loc_i+i,loc_j+j] - FocusValMean(focus_img,(loc_i+i,loc_j+j),window_size)) < sim_thresh:
        mask_img[loc_i+i,loc_j+j] = 1
        changed = True

  return changed

#region growth
def RegionGrow(img, canny_img,gaus_size = 11,sigma =1, focus_thresh = 0.5, sim_thresh = 0.5, seed_thresh = 0.95, edge_stop_thresh = 0.4,
                alpha = 1, beta = 1, gamma = 1, window_size = 11):


  gaus_kern = GaussianKernal(gaus_size, sigma)
  gaus_blur = ImgConvolve(img,gaus_kern)

  grad_img, grad_ang = ImgGrad(gaus_blur,'Sobel') 

  LoG_kernal = LoGKernal(gaus_size,sigma)
  LoG_img = ImgConvolve(img,LoG_kernal)

  

  #Normalize
  
  grad_img = grad_img - np.min(grad_img)
  grad_img = grad_img/np.max(grad_img)

  #for tenengrad
  #grad_img = np.square(grad_img)

  #LoG_img = LoG_img - np.min(LoG_img)
  LoG_img = np.abs(LoG_img)
  LoG_img = LoG_img/np.max(LoG_img)

  canny_img = canny_img - np.min(canny_img)
  canny_img = canny_img/np.max(canny_img)

  

  img_shape = np.shape(img)
  img_mask = np.zeros(img_shape)

  img_focus_score = np.zeros(img_shape)


  for i in range(np.shape(LoG_img)[0]):
    for j in range(np.shape(LoG_img)[1]):
  
      img_focus_score[i,j] = alpha*LoG_img[i,j] + beta*EdgeDensity(canny_img,(i,j),window_size) + gamma*grad_img[i,j]

  

  seed_thresh_val = np.max(img_focus_score) *seed_thresh


  max_loc = (0,0)



  for i in range(np.shape(LoG_img)[0]):
    for j in range(np.shape(LoG_img)[1]):
      if img_focus_score[i,j] >= seed_thresh_val:
        img_mask[i,j] = 1
        if img_focus_score[max_loc] < img_focus_score[i,j]:
          max_loc = (i,j)

  

  focus_thresh_val = focus_thresh * img_focus_score[max_loc]

  sim_thresh_val = sim_thresh * abs(img_focus_score[max_loc] - FocusValMean(img_focus_score,max_loc,window_size))

  not_fin = True

  

  checked_loc = np.zeros(img_shape)

  
  while not_fin:
    not_fin = False
    for i in range(1,img_shape[0]-1):
      for j in range(1,img_shape[1]-1):

        if img_mask[i,j] != 0 and checked_loc[i,j] == 0:
          checked_loc[i,j] = 1
          if np.sum(img_mask[i-1:i+2,j-1:j+2]) == 9:
            continue
          elif CheckNeighbour(img_focus_score,img_mask,focus_thresh_val,canny_img,sim_thresh_val,(i,j),window_size,edge_stop_thresh):
            not_fin = True
    

        

  
  return img_mask

def LocalVar(img,window_size,loc):

  start_i = loc[0]-math.floor(window_size/2)
  start_j = loc[1]-math.floor(window_size/2)

  img_shape = np.shape(img)

  end_i = loc[0]+math.floor(window_size/2)+1
  end_j = loc[1]+math.floor(window_size/2)+1

  if end_i>=img_shape[0]:
    end_i = img_shape[0]
  if end_j>=img_shape[1]:
    end_j = img_shape[1]


  loc_mean = np.sum(img[start_i:end_i,start_j:end_j])

  var = 0
  for i in range(start_i,end_i):
    for j in range(start_j,end_j):
      var += (img[i,j] - loc_mean)**2

  var = var / ((end_i-start_i) * (end_j-start_j))

  return var


def ImageDifference(img,derivative = 'interpolate'):

  img_shape = np.shape(img)

  img_der = np.zeros(img_shape)
  #uses abs since dont care about sign
  for k in range(1,img_shape[0]-1):

    if derivative == 'forward':
      img_der[k,:,:] ==  abs(img[k,:,:] - img[k+1,:,:])
    elif derivative == 'backward':
      img_der[k,:,:] ==  abs(img[k,:,:] - img[k-1,:,:])
    elif derivative == 'center':
      img_der[k,:,:] ==  abs(img[k+1,:,:] - img[k-1,:,:])
    elif derivative == 'interpolate':
      img_der[k,:,:] ==  abs(img[k+1,:,:] - img[k-1,:,:])

  return img_der



def FocusDiffernceMask(img, window_size = 11, kernel_mode = 4, derivative = 'backward',thresh = 0.5):
  #return focus back derivative currently
  img_shape = np.shape(img)
  print(img_shape)

  if kernel_mode == 4:
    lap_kernel = [[0,1,0],[1,-4,1],[0,1,0]]
  elif kernel_mode == 8:
    lap_kernel = [[1,1,1],[1,-8,1],[1,1,1]]
  else:
    lap_kernel = [[0,1,0],[1,-4,1],[0,1,0]]
  
  lap_img = np.zeros((img_shape[1],img_shape[2]))
  pad_w = math.floor(window_size/2)
  lap_var = np.zeros(img_shape)



  for k in range(img_shape[0]):
    lap_img = ImgConvolve(img[k,:,:],lap_kernel)
    lap_pad = np.pad(lap_img,pad_w,mode='reflect')
    for i in range(img_shape[1]):
      for j in range(img_shape[2]):
        lap_var[k,i,j] = np.var(lap_pad[i:(i+window_size),j:(j+window_size) ])

  var_dif_back = np.zeros(img_shape)
  #var_dif_for = np.zeros(img_shape)
  #var_dif_cen = np.zeros(img_shape)
  #dont care about sign on magnitude so use absolute val
  for k in range(1,img_shape[0]-1):
    
    var_dif_back[k,:,:] = abs(lap_var[k,:,:] - lap_var[k-1,:,:])
    
    #var_dif_for[k,:,:] = abs(lap_var[k,:,:] - lap_var[k+1,:,:])
    
    #var_dif_cen[k,:,:] = abs(lap_var[k+1,:,:] - lap_var[k-1,:,:])

    
  '''
  img_mask_back = np.zeros(img_shape,dtype=np.uint8)
  img_mask_for  = np.zeros(img_shape,dtype=np.uint8)
  img_mask_cen  = np.zeros(img_shape,dtype=np.uint8)

  for k in range(1,img_shape[0]-1):
    thresh_backward = np.max(var_dif_back[k,:,:])*thresh
    thresh_forward = np.max(var_dif_for[k,:,:])*thresh
    thresh_center = np.max(var_dif_cen[k,:,:])*thresh
    for i in range(img_shape[1]):
      for j in range(img_shape[2]):
        if var_dif_back[k,i,j] >= thresh_backward:
          img_mask_back[k,i,j] = 255
        if var_dif_for[k,i,j] >= thresh_forward:
          img_mask_for[k,i,j] = 255
        if var_dif_cen[k,i,j] >= thresh_center:
          img_mask_cen[k,i,j] = 255
  '''
 #return img_mask_back, img_mask_for, img_mask_cen
  return var_dif_back
  


def TenengradMask(img,canny_img,window_size = 11, alpha = 1, beta = 1, gamma = 1,seed_thresh = 0.9,grow_thresh = 0.6):

  img_shape = np.shape(img)

  img_grad, img_ang = ImgGrad(img,'Sobel')

  tenen_val = np.square(img_grad)

  #lap_kernel = [[0,1,0],[1,-4,1],[0,1,0]]
  lap_kernel = [[1,1,1],[1,-8,1],[1,1,1]]
  lap_img = ImgConvolve(img,lap_kernel)

  lap_var = np.zeros_like(img)
  pad_w = math.floor(window_size/2)
  lap_pad = np.pad(lap_img,pad_w,mode='reflect')

  for i in range(img_shape[0]):
    for j in range(img_shape[1]):
      lap_var[i,j] = np.var(lap_pad[i:(i+window_size),j:(j+window_size) ])
  
  
  #normalize
  lap_var = lap_var - np.min(lap_var)
  lap_var = lap_var / np.max(lap_var)

  tenen_val = tenen_val - np.min(tenen_val)
  tenen_val = tenen_val / np.max(tenen_val)

  focus_val = (alpha*tenen_val) + (beta*lap_var)

  canny_img = canny_img - np.min(canny_img)
  canny_img = canny_img * (1/np.max(canny_img))

  for i in range(img_shape[0]):
    for j in range(img_shape[1]):
      focus_val[i,j] = focus_val[i,j] + gamma*EdgeDensity(canny_img,(i,j),window_size)



  focus_val = focus_val - np.min(focus_val)

  focus_val = focus_val * (1/np.max(focus_val))


  img_mask = np.zeros_like(img)

  

  for i in range(img_shape[0]):
    for j in range(img_shape[1]):
      if focus_val[i,j] >= seed_thresh:
        img_mask[i,j] = 1

  not_fin = True

  check_loc = np.zeros_like(img)

  while not_fin:
    not_fin = False
    for i in range(img_shape[0]):
      for j in range(img_shape[1]):
        if img_mask[i,j] == 0 or check_loc[i,j] == 1:
          continue
        else:
          check_loc[i,j] = 1
          for ii in range(-1,2):
            if (i + ii >= img_shape[0]) or (i+ii < 0):
              continue
            for jj in range(-1,2):
              if (j + jj >= img_shape[1]) or (j + jj < 0):
                continue
              if focus_val[i+ii,j+jj] >= grow_thresh and img_mask[i+ii,j+jj] == 0:
                img_mask[i+ii,j+jj] = 1
                not_fin = True




  return img_mask


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
#doesnt work well
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

