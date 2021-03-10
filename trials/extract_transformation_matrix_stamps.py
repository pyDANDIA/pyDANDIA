import numpy as np
import os

def physical_from_transform_matrix(transform_matrix):
    #Definitions here https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.ProjectiveTransform
    
    scale_x = (transform_matrix[0, 0] ** 2 + transform_matrix[1, 0] ** 2) ** 0.5
    scale_y = (transform_matrix[0, 1] ** 2 + transform_matrix[1, 1] ** 2) ** 0.5
    
    angle = np.arctan2(transform_matrix[1,0],    transform_matrix[0,0])
    shear = np.arctan2(-transform_matrix[0, 1], transform_matrix[1, 1]) - angle
    
    shift_x = transform_matrix[0,2]
    shift_y = transform_matrix[1,2]


    return scale_x,scale_y,angle,shear,shift_x,shift_y
    
    
directory_reduction = '/data02/rstreet/data_reduction/ROME-FIELD-01/ROME-FIELD-01_lsc-doma-1m0-05-fa15_ip/'
#directory_reduction = '/home/etienne/Work/Photometry/Gaia19dke/MUSCAT_ip/'
directory_resampled = directory_reduction+'resampled/'

number_of_stamps = 16

all_frames_resampled = [i for i in os.listdir(directory_resampled)]

for frame in all_frames_resampled:
    image_matrix =  np.load(directory_resampled+frame+'/warp_matrice_image.npy')
    scalex,scaley,angle,shear,sx,sy = physical_from_transform_matrix(image_matrix)
    
    print(frame,angle,scalex,scaley,angle,shear,sx,sy) #sx and sy should be very close(but not exact) to the one in the metadata, except sign -1
    
    for i in range(number_of_stamps):

        
        stamp_name = '/warp_matrice_stamp_'+str(i)+'.npy'
        
        stamp_matrix = np.load(directory_resampled+frame+stamp_name)
        sscalex,sscaley,sangle,sshear,ssx,ssy = physical_from_transform_matrix(stamp_matrix)
        #print(stamp_name,sscalex,sscaley,sangle,sshear,ssx,ssy)
        parameters = stamp_matrix.ravel() # I propose to store the parameters like this, then we can reconstruct using .reshape(3,3)
import pdb; pdb.set_trace()
