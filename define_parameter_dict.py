#This file defines dictionary of parameters to be used globally.
import cv2

#Input parameters
params_dict = {}
params_dict['car_dir'] = './data/vehicles/'
params_dict['non_car_dir'] = './data/non-vehicles/'
params_dict['input_video_file'] = 'project_video.mp4'

#Other parameters
params_dict['color_space'] = 'yuv'
params_dict['svm_kernel'] = 'linear'
params_dict['detection_scales'] = [.75, 1, 2]
params_dict['consecutive_number_of_frames'] = 3 #The number of frames in which a detection box must appear consecutively.


##########################  Derive rest of the parameters based on above choices   #####################
#Video frames are by default in RGB space.  The images we read through cv2 are in BGR space.  Hence prepare transformation paramters accordingly
csp = params_dict['color_space']
svm_pickle_file = './svm_'
svm_pickle_file = svm_pickle_file+csp+'_'
if csp == 'hsv':
    params_dict['cvtTransformStill'] = cv2.COLOR_BGR2HSV
    params_dict['cvtTransformVideoFrame'] = cv2.COLOR_RGB2HSV
elif csp=='hls':
    params_dict['cvtTransformStill'] = cv2.COLOR_BGR2HLS
    params_dict['cvtTransformVideoFrame'] = cv2.COLOR_RGB2HLS
elif csp=='luv':
    params_dict['cvtTransformStill'] = cv2.COLOR_BGR2LUV
    params_dict['cvtTransformVideoFrame'] = cv2.COLOR_RGB2LUV
elif csp=='rgb':
    params_dict['cvtTransformStill'] = cv2.COLOR_BGR2RGB
    params_dict['cvtTransformVideoFrame'] = [] #No need to change because already in RGB space
elif csp=='yuv':
    params_dict['cvtTransformStill'] = cv2.COLOR_BGR2YUV
    params_dict['cvtTransformVideoFrame'] = cv2.COLOR_RGB2YUV
elif csp=='YCrCb':
    params_dict['cvtTransformStill'] = cv2.COLOR_BGR2YCrCb
    params_dict['cvtTransformVideoFrame'] = cv2.COLOR_RGB2YCrCb
    
svm_pickle_file = svm_pickle_file+params_dict['svm_kernel'] + '.pickle'

params_dict['svm_pickel_file_for_video'] = svm_pickle_file
params_dict['svm_pickle_file'] = svm_pickle_file
params_dict['out_video_file'] = params_dict['input_video_file'].rsplit('.',1)[0] + '_out_' + csp + '_' + params_dict['svm_kernel'] + '.' + params_dict['input_video_file'].rsplit('.',1)[1]

