from HoG_SVM_SlidingWindow import *
from define_parameter_dict import params_dict
from moviepy.editor import VideoFileClip
import numpy as np

with open(params_dict['svm_pickel_file_for_video'],'rb') as fp:
    [cl_svm_vid, std_scaler_vid] = pickle.load(fp)
scale_list = params_dict['detection_scales']
consecutive_appearance = params_dict['consecutive_number_of_frames']

#This function takes two rectangles R1 and R2.  Returns ratio of areas of intersection of R1,R2 to the area of R1
def rect_overlap(rect1, rect2):
    overlap = 0.0
    row_top = max(rect1[0], rect2[0])
    row_bottom = min(rect1[1], rect2[1])
    col_left = max(rect1[2], rect2[2])
    col_right = min(rect1[3], rect2[3])
    if col_right>col_left and row_bottom>row_top:
        overlap = (col_right-col_left)*(row_bottom-row_top)/(rect1[1]-rect1[0])/(rect1[3]-rect1[2])
    return overlap
        
    

def process_video_frame(img):
    process_video_frame.counter += 1
    if params_dict['cvtTransformVideoFrame']:
        img_process = cv2.cvtColor(img, params_dict['cvtTransformVideoFrame'])
    else:
        img_process = np.copy(img)
    car_boxes = find_cars(img_process, scale_list, cl_svm_vid, std_scaler_vid)
    #res_image = add_result_thumbnail(img, car_boxes)
    consolidated_boxes_cur_frame = detect_and_consolidate(img, car_boxes)
    retain_box = []
    if len(process_video_frame.prev_boxes) >= consecutive_appearance:
        for box in consolidated_boxes_cur_frame: #For each box in the curret frame
            retain_box.append(True)
            for prev_box_list in process_video_frame.prev_boxes: #Get the list of boxes in one of the previous frames
                max_overlap = 0
                for prev_box in prev_box_list: #For each box in the previous frame
                    overlap = rect_overlap(box, prev_box)
                    if overlap>max_overlap:
                        max_overlap = overlap
                if max_overlap<.5:
                    retain_box[-1] = False
    process_video_frame.prev_boxes.append(consolidated_boxes_cur_frame)
    if len(process_video_frame.prev_boxes) > consecutive_appearance:
        process_video_frame.prev_boxes.pop(0)
        
    consistent_boxes = []
    if retain_box:
        consistent_boxes = [consolidated_boxes_cur_frame[i] for i in range(len(consolidated_boxes_cur_frame)) if retain_box[i]]
    res_image = np.copy(img)
    for box in consistent_boxes:
        cv2.rectangle(res_image, (box[2], box[0]), (box[3], box[1]), color=(255, 0, 0), thickness=6)
    return res_image

input_video_file = params_dict['input_video_file']
out_video_file = params_dict['out_video_file']
in_clip = VideoFileClip(input_video_file)
process_video_frame.counter = 0
process_video_frame.prev_boxes = []
out_clip = in_clip.fl_image(process_video_frame)
out_clip.write_videofile(out_video_file, audio=False)

