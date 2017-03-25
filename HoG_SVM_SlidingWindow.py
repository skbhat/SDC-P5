from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import svm
from skimage.feature import hog
from scipy.ndimage.measurements import label as label_pixel
import cv2
import numpy as np

import glob
import time
import os
import pickle
import matplotlib.pyplot as plt

from define_parameter_dict import params_dict

def load_image(file_name, params_dict):
    im = cv2.imread(file_name)
    if params_dict['cvtTransformStill']:
        im = cv2.cvtColor(im, params_dict['cvtTransformStill'])
    return im

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

def extract_feature(img):
    combined_feature = []
    combined_feature.append(bin_spatial(img))
    combined_feature.append(color_hist(img))
    hog_feature = np.concatenate((get_hog_features(img[:,:,0]), get_hog_features(img[:,:,2]), get_hog_features(img[:,:,2])))
    combined_feature.append(hog_feature)    
    return np.concatenate(combined_feature)

def compute_feature_for_list(im_file_list):
    feature_list = []
    for i in range(len(im_file_list)):
        im = load_image(im_file_list[i], params_dict)
        feature = extract_feature(im)
        feature_list.append(feature)
    
    return feature_list


                
def train_svm(car_dir, non_car_dir, kernel='linear'):
    
    t_start = time.time(); print('\ntrain_svm_on_hog : Listing files')
    cars = [file for file in glob.glob(car_dir + '/**/*.png', recursive=True)]
    non_cars = [file for file in glob.glob(non_car_dir + '/**/*.png', recursive=True)]
    #cars = cars[:10]
    #non_cars = non_cars[:10]
    t_end = time.time(); print('train_svm_on_hog : Finished in ' + str(round(t_end-t_start,1)) + ' seconds')
    
    
    t_start = time.time(); print('\ntrain_svm_on_hog : Computing hog features of ' + str(len(cars)) + ' car images')
    car_vecs = compute_feature_for_list(cars)
    print('train_svm_on_hog : Computing hog features of ' + str(len(non_cars)) + ' non_car images')
    non_car_vecs = compute_feature_for_list(non_cars)
    t_end = time.time(); print('train_svm_on_hog : Finished in ' + str(round(t_end-t_start,1)) + ' seconds')
    
    t_start = time.time(); print('\ntrain_svm_on_hog : Preparing training data')
    all_vecs = np.vstack((car_vecs, non_car_vecs))
    std_scaler = StandardScaler().fit(all_vecs)  # per-column scaler
    all_vecs = std_scaler.transform(all_vecs)
    all_labels = np.concatenate((np.ones(len(car_vecs)), np.zeros(len(non_car_vecs))))
    train_vecs, test_vecs, train_labels, test_labels = train_test_split(all_vecs, all_labels, test_size=0.1)
    t_end = time.time(); print('train_svm_on_hog : Finished in ' + str(round(t_end-t_start,1)) + ' seconds')

    t_start = time.time(); print('\ntrain_svm_on_hog : training SVM')    
    if kernel == 'linear':
        cl_svm = LinearSVC()
    else:
        cl_svm = svm.SVC(kernel=kernel)
    cl_svm.fit(train_vecs, train_labels)
    t_end = time.time(); print('train_svm_on_hog : Finished in ' + str(round(t_end-t_start,1)) + ' seconds')
    print('Validation accuracy of cl_svm = ', round(cl_svm.score(test_vecs, test_labels), 4))
    
    return cl_svm, std_scaler

def get_svm_scaler(car_dir, non_car_dir, svm_pickle_file, use_trained_svm):
    if use_trained_svm and os.path.isfile(svm_pickle_file):
        print('SVM is already trained.  Loading ...')
        with open(svm_pickle_file,'rb') as fp:
            [cl_svm, std_scaler] = pickle.load(fp)
    else:
        print('Need to train SVM')
        
        cl_svm, std_scaler =train_svm(car_dir, non_car_dir, kernel=params_dict['svm_kernel'])
        with open(svm_pickle_file,'wb') as fp:
            pickle.dump([cl_svm, std_scaler],fp)
            
    return cl_svm, std_scaler
                                         
def find_cars_at_specified_scale(img, scale, svc, X_scaler, orient=9, pix_per_cell=8, cell_per_block=2):
    
    #draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    ystart=400
    ystop=610
    if scale<1: #When scale is 2, ystop=610.  At the minimum scale 1 it must be 460.  So applying linearly.
        ystop = 460 + (scale-1.0)*(610-460)/(2.0-1.0)
    
        
        
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = np.copy(img_tosearch) #convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 4  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    potential_box_list = [] #List of image patches on which SVM classifies as car
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg)
            hist_features = color_hist(subimg)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = [ytop_draw+ystart, ytop_draw+win_draw+ystart, xbox_left, xbox_left+win_draw]
                potential_box_list.append(box)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
    #return draw_img
    return potential_box_list

def find_cars(img, scale_list, svc, std_scaler):
    car_boxes = []
    
    for scale in scale_list:
        car_boxes = car_boxes + find_cars_at_specified_scale(img, scale, svc, std_scaler)
    return car_boxes

def generate_heatmap(img, box_list):
    heat_map = np.zeros((img.shape[0], img.shape[1]),dtype = np.float32)
    for box in box_list:
        heat_map[box[0]:box[1], box[2]:box[3]]+=1
    #heat_map = np.clip(heat_map, 0, 255)
    heat_map = heat_map*255/heat_map.max()
    return heat_map.astype(np.uint8)

def add_result_thumbnail(img, box_list):
    thumb_scale = 3.0
    ret_im = np.copy(img)
    heat_map = generate_heatmap(img, box_list)
    labels = label_pixel(heat_map)
    
    #Mention the number of cars detected
    put_text = 'cars : ' + str(labels[1])
    cv2.putText(ret_im, put_text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2,  lineType = cv2.LINE_AA)
    
    #Put heatmap thumbnail
    heat_map_thumb = cv2.resize(heat_map, (np.int(img.shape[1] / thumb_scale), np.int(img.shape[0] / thumb_scale)))
    ret_im[20:20+heat_map_thumb.shape[0], 200:200+heat_map_thumb.shape[1], 0] = heat_map_thumb
    ret_im[20:20+heat_map_thumb.shape[0], 200:200+heat_map_thumb.shape[1], 1] = 0
    ret_im[20:20+heat_map_thumb.shape[0], 200:200+heat_map_thumb.shape[1], 2] = 0
    
    #Draw all candidate boxes
    for box in box_list:
        cv2.rectangle(ret_im,(box[2], box[0]),(box[3],box[1]),(0,0,180),thickness=2)
        
    #Draw consolidated boxes
    for i in range(1,labels[1]+1):
        a, b = np.where(labels[0] == i)
        x_min, y_min = np.min(b), np.min(a)
        x_max, y_max = np.max(b), np.max(a)
        cv2.rectangle(ret_im, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=6)
    
    
    
    return ret_im

def detect_and_consolidate(img, box_list):
    
    heat_map = generate_heatmap(img, box_list)
    heat_map[heat_map<2] = 0
    labels = label_pixel(heat_map)
        
    #Consolidate the boxes
    ret_boxes = []
    for i in range(1,labels[1]+1):
        a, b = np.where(labels[0] == i)
        x_min, y_min = np.min(b), np.min(a)
        x_max, y_max = np.max(b), np.max(a)
        ret_boxes.append([y_min, y_max, x_min, x_max])
    
    
    
    return ret_boxes

###############################  Main Process   ###########################
if __name__ == '__main__':
    #Obtain SVM classifier
    use_trained_svm = True        
    svm_pickle_file = params_dict['svm_pickle_file']
    car_dir = params_dict['car_dir']
    non_car_dir = params_dict['non_car_dir']
    cl_svm, std_scaler = get_svm_scaler(car_dir, non_car_dir, svm_pickle_file, use_trained_svm) #Load the existing SVM if available.  Otherwise compute it by training
    
    #Run on test images
    t_start = time.time(); print('\n Detecting cars in test images ')
    test_im_list = glob.glob('./test_images/*.jpg')
    scale_list = params_dict['detection_scales']
    count = 0;
    for test_im_name in test_im_list:
        count = count+1
        img = load_image(test_im_name, params_dict)
        car_boxes = find_cars(img, scale_list, cl_svm, std_scaler)
        img = cv2.imread(test_im_name); img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        res_image = add_result_thumbnail(img, car_boxes)
        out_fname = './output_images/' + test_im_name.rsplit('/',1)[1]
        res_image = cv2.cvtColor(res_image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_fname, res_image)
    t_end = time.time(); print('Took ' + str(round(t_end-t_start,1)) + ' seconds to process ' + str(count) + ' images at ' + str(len(scale_list)) + ' scales')
    
    
    
    
    print('Over')