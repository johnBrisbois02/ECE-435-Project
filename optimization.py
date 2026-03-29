
from PIL import Image
import numpy as np
import tifffile
from functions import ImgGrad, ImgConvolve, ImageDifference
import pandas as pd
import math
import pickle
from sklearn.model_selection import train_test_split
class Batch:

    def __init__(self,data,truth,batch_num):
        self.data = data
        self.truth = truth
        self.batch_num = batch_num
        self.name = "batch_" + str(batch_num)
        
        self.num_image = np.shape(data)[0]

        
        self.loc_var = np.zeros_like(data)
        self.tenengrad = np.zeros_like(data)
        self.var_dif = np.zeros_like(data)
        self.laplacian = np.zeros_like(data)
        self.grad_ang = np.zeros_like(data)

    def save_file(self):
        
        filename = self.name + ".dat"
        with open(filename, "wb") as f:
            pickle.dump(self,f)

#allows skipping the computation time on inputs
def BatchLoad(filename):

    with open(filename,"rb") as f:
        return pickle.load(f)

    

def GetMatrices(cur_batch,window_size=11):

    lap_kernel = [[1,1,1],[1,-8,1],[1,1,1]]
    pad_w = math.floor(window_size/2)

    img_stack = cur_batch.data

    img_shape = np.shape(img_stack)

    lap_mat = np.zeros(img_shape)
    tenen_mat = np.zeros(img_shape)
    var_dif_mat = np.zeros(img_shape)
    loc_var_mat = np.zeros(img_shape)
    grad_ang = np.zeros(img_shape)
    

    for k in range(img_shape[0]):

        laplacian = ImgConvolve(img_stack[k,:,:],lap_kernel)
        lap_mat[k,:,:] = laplacian
        lap_pad = np.pad(laplacian,pad_w,mode='reflect')

        tenen_mat[k,:,:], grad_ang[k,:,:] = ImgGrad(img_stack[k,:,:], mode = 'Sobel', val_mode = 'Tenengrad')
        

        for i in range(img_shape[1]):
            for j in range(img_shape[2]):
                loc_var_mat[k,i,j] = np.var(lap_pad[i:(i+window_size),j:(j+window_size) ])

    var_dif_mat = ImageDifference(loc_var_mat,derivative = 'interpolate')

    
    cur_batch.loc_var = loc_var_mat
    cur_batch.tenengrad = tenen_mat
    cur_batch.var_dif = var_dif_mat
    cur_batch.laplacian = lap_mat
    cur_batch.grad_ang = grad_ang


    return 

def DataSets(batches):

    rows = []

    for batch in batches:
        for i in range(batch.num_image):

            rows.append({
                "batch_num": batch.batch_num,
                "sample_idx": i
            })

    df = pd.DataFrame(rows)

    train_df,test_df = train_test_split(
        df,
        test_size = 0.2,
        random_state=42,
        stratify=["batch_num"]
    )
    

    return train_df,test_df

def Train(train_df, test_df):

    weights = np.ones(6)

    reg_lambda = 1


    return weights

def main():

    #batch2, batch3, batch6, batch7 = BatchFormat()

    #batch2.save_file()
    #batch3.save_file()
    #batch6.save_file()
    #batch7.save_file()

    batch2 = BatchLoad("batch_2.dat")
    batch3 = BatchLoad("batch_3.dat")
    batch6 = BatchLoad("batch_6.dat")
    batch7 = BatchLoad("batch_7.dat")

    batch_array = [batch2,batch3,batch6,batch7]

    train_df, test_df = DataSets(batch_array)

    


    return



def BatchFormat():

    window_size = 15
    
    data_stack_2 = tifffile.imread('Project_Data/2_image.tiff')
    truth_stack_2 = tifffile.imread('Project_Data/2_mask.tiff')
    batch_2 = Batch(data_stack_2,truth_stack_2,2)
    GetMatrices(batch_2,window_size)

    data_stack_3 = tifffile.imread('Project_Data/3_image.tiff')
    truth_stack_3 = tifffile.imread('Project_Data/3_mask.tiff')
    batch_3 = Batch(data_stack_3,truth_stack_3,3)
    GetMatrices(batch_3,window_size)

    data_stack_6 = tifffile.imread('Project_Data/6_image.tiff')
    truth_stack_6 = tifffile.imread('Project_Data/6_mask.tiff')
    batch_6 = Batch(data_stack_6,truth_stack_6,5)
    GetMatrices(batch_6,window_size)

    data_stack_7 = tifffile.imread('Project_Data/7_image.tiff')
    truth_stack_7 = tifffile.imread('Project_Data/7_mask.tiff')
    batch_7 = Batch(data_stack_7,truth_stack_7,7)
    GetMatrices(batch_7,window_size)

    

    return batch_2, batch_3, batch_6, batch_7


main()



'''
data to get
local variance of laplacian (window around the pixel)
difference (forward  or back) of laplacian variance between neighbour slices ()
magnitude of gradient squared (tenengrad)
total variance

stuff to add if poor performance
laplacian of gaussian (LoG)
Brenner
other focus measures
'''