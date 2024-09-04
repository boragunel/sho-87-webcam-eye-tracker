
import pandas

### you need to create a detector OpenCV 
# to retrieve frames from the webcam. 
# By itself, reading from the webcam is quite straightforward:

import cv2
capture = cv2.VideoCapture(0) # names capture variable as the default camera (if multiple cameras, they are referred to as 1,2 or more ...)
capture.read()                # 'read' method tries to grab frame from camera

print(capture.read())

#reading web cam frames is a blocking action as reading 
#having to read every single time slows down the process
#you need a code that continuously process webcam image tensors
#below code shows a thread that continuously reads from webcam
#and puts the frame from that queue and porcess however we need.



import queue
import threading

class Detector:
    def __init__(self):
        # Threaded webcam capture
        self.capture = cv2.VideoCapture(0) #initialize capture object as default camera
        self.q = queue.Queue()             #creates queue to store frames captured from camera
        t = threading.Thread(target=self._reader)   #Create a queue to store frames captured from the camera
        t.daemon = True                             #start a new thread to read frames from the camera
        t.start()                                   #starts thread
    def _reader(self):
        while True:
            ret, frame = self.capture.read()    #ret is successful capture, frame represents frame tensors, read() method provided by openCV
            if not ret:                         #breaks cycle when image is unsuccessful
                break
            if not self.q.empty():              #Remove oldest frame from queue without blocking
                try:                            
                    self.q.get_nowait()         #Except if the queue is already empty
                except queue.Empty:
                    pass
            self.q.put(frame)                   #Add most recent frame to the queue
    def get_frame(self):                        
        frame = self.q.get()                    #Getting most recent frame
        # ... process the frame here ...        #further codes here can also allow you to process the image by using filters
    def close(self):                            
        self.capture.release()                  #self.capture.release()


#We need to make decisions on what features will be useful. we are not yet using an existing dataset
#we have flexibility in terms of what inputs we can pass into our model
#but what features are useful in this case?

#Face image:
    ##At the bare minimum, we need to perform face detection to isolate face region
    ##Is a single face image enough?
    ##CNN's can learn image features, and combinations of those features,
    ##so on one hand a single face image would
    ##technically contain everythin we possibly need.
    ##Problems presented however: 
    # 1) webcam images don’t have the highest resolution, and 
    # 2) deep models are expensive to run in real time.

###Eye images:
            #As we're creating an eye traker, it might be useful to extract
            #images of just the eye regions so we can focus on the model on those areas
            #these could be used instead of, or in conjuction with, the full face image.
            #Passing a face image and 2 eye images might allow us to run a shallower
            #model while still empathysing areas that we think might be the most important

###Head and angle positions:
            #Head angle might be necessary for eye movements as it interacts with
            #eye movements. If eyes are looking right but face is turned left,
            #The person could still be looking at the centre but this will not be reigstered by the machine
            #know unless head angle is tracked.

###to summarize, the input will include 5 channels:
            #1. unaligned faces
            #2. left eye
            #3. right eye
            #4.Aligned face:
                #4.1 Head position
                #4.2 head angle

###Face detection:
    ###The first step into the process is to isolate the face in the webcam image.
    ###most common method is Haar classifier thechnique (available as a python package)
    ###These methods however inaccurate sometimes (thus unreliable)
    ###especially when the face is not looking directly to the camera
    ###Face angle is an issue when in the case of eye tracker, the angle can
    ###get quite extreme if the user is close to the monitor and looking at one
    ###of the screen corners.
    ###There are many other alternative deep-learning methods or techniques for face
    ###detection. In this case we'll use the CNN face detector that is
    ###available in the dlib package. pre-trained CNN weights can be found either on 
    ###the dlib website or in python project: https://github.com/sho-87/webcam-eye-tracker/tree/master/trained_models
    ###What they did was simply create an instance of face detector and load with
    ###pre-trained weights
    ###Then they can get a frame from the webcam and pass it to the detector
    ###CNN dlib returns a list of rectangles, which contain the coordinates
    ###of the square region around each detected face:



import dlib
class Detector:
    def __init__(self):
        ...
        self.detector = dlib.cnn_face_detection_model_v1("trained_models/mmod_human_face_detector.dat") #create instance of detector and load with pre-trained weights
    def get_frame(self):
        frame = self.q.get()                    ##repeat procedure by getting a frame
        dets = self.detector(frame, 0)          ##passes values to detector

        ##dlib CNN detector returns a list of rectangles which contains
        ##coordinates of the square region around each detected face.


##we can use those coordinates to crop out a region of the webcam frame
##for a face image. This basic face image will serve as one of the possible inputs
##to our eye tracker.

##Facial landmark detection:
    ##Many ways of performing full landmark detection, for this we only need
    ##4 eye corners
    ##dlib comes with 5-point landmark detector (eye coreners and nose), will help us get the eye coordinates
    ##We can create a shape predictor object and pass it the trained weights

from collections import OrderedDict
from imutils.face_utils import shape_to_np

class Detector:
    def __init__(self):
        ...
        self.landmark_idx = OrderedDict([("right_eye", (0, 2)), ("left_eye", (2, 4))])                      ##maps the name of facial features
                                                                                                            ##to corresponfing regions
        self.detector = dlib.cnn_face_detection_model_v1("trained_models/mmod_human_face_detector.dat")     ##Load pre-trained weights for face detection model from dlib
        self.predictor = dlib.shape_predictor("trained_models/shape_predictor_5_face_landmarks.dat")        ##Load pre-trained weights for facial landmark predictor from dlib 
    def get_frame(self):
        frame = self.q.get()
        dets = self.detector(frame, 0)                                                                      ##gets thbe frame and loads it to detector
        if len(dets) == 1:                                                                                  ##makes sure only one face is detected                                                                          ##
            # Get feature locations     
            features = self.predictor(frame, dets[0].rect)                                                  ##uses shape predictor to detect facial landmarks on the present frame
            reshaped = shape_to_np(features)                                                                ##Converts tensor into numpy array
            l_start, l_end = self.landmark_idx["left_eye"]                                                  ##extracts start and end indices for left eye
            r_start, r_end = self.landmark_idx["right_eye"]                                                 ##extracts start and end indices for right eye
            l_eye_pts = reshaped[l_start:l_end]                                                             ##slices re-shaped numpy to get coordinates of left eye landmark
            r_eye_pts = reshaped[r_start:r_end]                                                             ##slices re-shaped numpy to get coordinates of right eye landmark
            l_eye_center = l_eye_pts.mean(axis=0).astype("int")                                             ##Positional estimate of the centre of the left eye
            r_eye_center = r_eye_pts.mean(axis=0).astype("int")
            l_eye_width = l_eye_pts[1][0] - l_eye_pts[0][0]
            r_eye_width = r_eye_pts[0][0] - r_eye_pts[1][0]
            l_eye_img = frame[l_eye_center[1] - int(l_eye_width / 2) : l_eye_center[1] + int(l_eye_width / 2),
                            l_eye_pts[0][0] : l_eye_pts[1][0]]
            r_eye_img = frame[r_eye_center[1] - int(r_eye_width / 2) : r_eye_center[1] + int(r_eye_width / 2),
                            r_eye_pts[1][0] : r_eye_pts[0][0]] 
                                                                    ###The above allows you to have square images of each eye
                                                                    ###This can be used as eye tracker inputs


##We can calculate the eye width as the distance between the eye corners, and use that width as the 
# resulting image height to keep the image square:


## The above allows you to have square images of each eye 
# that can be used as eye tracker inputs








