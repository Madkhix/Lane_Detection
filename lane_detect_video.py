import numpy as np
import cv2
from skimage.transform import resize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model

# Load Module
model = load_model('full_CNN_model.h5')

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):

    # Preparing Images For The Model
    small_img = resize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Neural Network Prediction (Normalized By Multiplication)
    prediction = model.predict(small_img)[0] * 255

    # Add Prediction To Get Average
    lanes.recent_fit.append(prediction)
    # Son beş tahmin kullanılır
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Average Detection Is Calculated
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Produced In Red And Blue Color Sizes, Stored In Green Color
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Matched With Original Image
    lane_image = resize(lane_drawn, (720, 1280, 3))

    # The Strip Drawing Is Merged With The Original Image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0, dtype=cv2.CV_32F)

    return result

lanes = Lanes()

# işlenmiş video
vid_output = 'test4_out.mp4'

# giriş videosu
clip1 = VideoFileClip("test4.mp4")

vid_clip = clip1.fl_image(road_lines)
vid_clip.write_videofile(vid_output, audio=False)
