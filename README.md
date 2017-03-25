# Vehicle Detection Project

The aim of this project is to perform vehicle detection in a video.  I have used the dataset provided along with the tutorial, which essentially contains a set of car and non-car images.  Hence the final detector will detect cars in a given image

## Methodology
### Features
As suggested in the tutorial, a combination of HoG, Histogram and raw pixel from the image are used to build the feature vector.  This combination provides info from gradient, color distribution and texture.
### Training
Training feature vectors extracted from +ve (car images) and -ve (non-car images) samples are used train a two class classifier.  I use SVM with linear kernel.  I have tried RBF also.  But it was very slow (due to huge number of support vectors).
### Testing on single image
Given a test image, sliding windows at multiple-scales are scanned across it.  Each window is classified using the trained classifier.  Each +ve window gives one vote to the pixel within.  Voted regions are lableled using tool scipy.ndimage.measurements.  Each region with same label are grouped to produce consolidated rectangular boxes.  These boxes indicate the regions in the image potentially containing cars.
### On video
Each frame of the video is passed through the test stage.  An additional condition is used to remove false positives.  For consolidated box detected in the current frame is compared with previous n (in my experiment n=3) frames.  If 50% of the region of this box turn out to be +ve in those n frames, then the detection is said to be consistent and displayed in the video.

## Code discription
There are 3 python script files
1. [`HoG_SVM_SlidingWindow.py`](HoG_SVM_SlidingWindow.py)
2. [`process_video.py`](process_video.py)
3. ['define_parameter_dict.py'](define_parameter_dict.py)
