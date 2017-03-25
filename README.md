# Vehicle Detection Project

The aim of this project is to perform vehicle detection in a video.  I have used the dataset provided along with the tutorial, which essentially contains a set of car and non-car images.  Hence the final detector will detect cars in a given image

## Methodology
### Features
As suggested in the tutorial, a combination of HoG, Histogram and raw pixel from the image are used to build the feature vector.  This combination provides info from gradient, color distribution and texture.
### Training
Training feature vectors extracted from +ve (car images) and -ve (non-car images) samples are used train a two class classifier.  I use SVM with linear kernel.  I have tried RBF also.  But it was extremely slow during runtime (due to huge number of support vectors).
### Testing on single image
Given a test image, sliding windows at multiple-scales are scanned across it.  Each window is classified using the trained classifier.  Each +ve window gives one vote to the pixel within.  Voted regions are lableled using tool scipy.ndimage.measurements.  Each region with same label are grouped to produce consolidated rectangular boxes.  These boxes indicate the regions in the image potentially containing cars.
### On video
Each frame of the video is passed through the test stage.  An additional condition is used to remove false positives.  For consolidated box detected in the current frame is compared with previous n (in my experiment n=3) frames.  If 50% of the region of this box turn out to be +ve in those n frames, then the detection is said to be consistent and displayed in the video.

## Code discription
There are 3 python script files
1. [`HoG_SVM_SlidingWindow.py`](HoG_SVM_SlidingWindow.py)
2. [`process_video.py`](process_video.py)
3. [`define_parameter_dict.py`](define_parameter_dict.py)

[`HoG_SVM_SlidingWindow.py`](HoG_SVM_SlidingWindow.py).  Contains the portion needed for feature selection, training and running the sliding window search on the images provided in [`test_images`](test_images).  Once trained the trained SVM classifier will be stored in a pickle file and next time it will be loaded (without running training again).  The function *find_cars()*, is the wrapper for performing sliding window search at multiple scales.  It invokes the function *find_cars_at_specified_scale()* for a particular scale, to scan the sliding windows at a particular scale and identify the locations where the classifier response is +ve.  Once we have the list of boxes at which the classifier response is positive, we cause invoke *add_result_thumbnail()* to generate heatmap, group the regions by labeling and consolidate the final rectangular locations where we have high confidance.  After finishing these computations, *add_result_thumbnail()*, annotates the image with the result (all boxes in blue, consolidated boxes in red) and thumbnail of heatmap.  For images in [`test_images`](test_images) results are stored in [`output_images`](output_images)

Using [`process_video.py`](process_video.py), we can perform the above detection on all frames of a video.  It has a function *process_video_frame()* which is invoked on each frame.  A variable *consecutive_appearance* sets the number of frames in which a box need to appear consecutively in order to be consistent.  This is used to remove fals positives.  If at least 50% of a consolidated box in the current frame is found out to be +ve in those previous frames, it is declared as consistent and plotted in the final output video.

Finally, [`define_parameter_dict.py`](define_parameter_dict.py) is the file in which parameters are set for global usage (i.e. in the other two scripts).  The first 15 lines of this file defines all the input and learning parameters.  You can specify the training dataset location, the input video for [`process_video.py`](process_video.py) etc.  One can choose the color space, svm kernel, scales at which slinding window search need to be performed etc.  For YUV color space, and linear svm kernel, the trained SVM will be stored in [`svm_yuv_linear.pickle`](svm_yuv_linear.pickle).  Once trained, the same file will be used (no re-training), for that combination of parameters.  For retraining, please delete this output file.
