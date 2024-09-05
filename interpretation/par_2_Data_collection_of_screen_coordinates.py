

###Choosing and Collecting data!!!

##We need a way to get labels
##We need a semi-efficient data collection method to get our X-Y coordinate

##We cannot collect too much redundant or repeated data because that will inflate our data size and make training longer
##The plan:

##The basic idea is quite simple: If we render (displaying images) a target image
##at a known location, then we can record the XY coordinate to act
##as a "label"

##2-part application for data colleciton:
    #Calibration mode:
            #Objective is to calibrate so to make sure device accurately detects where you're looking at
            #Screen has 9 key points: 1 centre, 4 corners, 4 midpoints of those corners
            #program renders 9 targets on those locations and capture when the user gaze on those regions
            #Ensures that most important locations are captured
    #Data collection mode:
        #Render a moving target on the screen, then every few milliseconds save X-Y coordinates
        #and webcam features simultaneously. Want to paint the screen as much as possible
        #within the shortest amount of time.

print('bora')

import pygame
from Gaze import Detector
from utils import get_config
import random


pygame.init()                                                     #Initizalize all pygame modules
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)       #Creates window that takes up screen, resolution automatically set to match displays resolution
SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")      #Loads configuration settings
detector = Detector(output_size=SETTINGS["image_size"])           #Creates instance of Detector class from Gaze module which sets up an eye traking or face detection system
                                                                  #Specifies image size to dettector which originate from settings
clock = pygame.time.Clock()                                       #Creates clock that is used to control frame rate
bg = random.choice((COLOURS["black"], COLOURS["gray"]))           #Selects between colours black and gray (from COLOUR dictionary)
'''
while True:                                                       #continuously runs the program until stopped manually
    screen.fill(bg)                                                           #Fills entire screen with selected background colour
    l_eye, r_eye, face, face_align, head_pos, angle = detector.get_frame()    #get input frame tensor values from 6 inputs listed below and discussed in previous section
    # ... do things here ...
    ticks = clock.tick(SETTINGS["record_frame_rate"])             #Limits frame rate of the loop to the value specified in SETTINGS dictionary
    pygame.display.update()                                       #Updates the screen filling with different colours

'''

#We need a target class which is an object that we can render itself at specific screen coordinates
#and can moce in straight line to new target locations
#Issue: "It is important that we also control the speed of the moving target"
#However because webcam features are calculated on the main thread, there may be times where frame
#rate drops, which ends up giving the target variable speed
#in the

class Target:                                                                       
    def __init__(self, pos, speed, radius=10, color=(255, 255, 255)):                    #Initizalize Target object with position, speed, raidus and color
        super().__init__()                                                               
        self.x = pos[0]                                                                  #Initial x position of target (dimension [0])
        self.y = pos[1]                                                                  #Initialize y position of target (dimension [1])
        self.speed = speed                                                               #defines the target speed
        self.radius = radius                                                             #size of the target
        self.color = color                                                               #color of the target
        self.moving = False                                                              #Defines whether the object is in moving or resting state
    def render(self, screen):
        pygame.draw.circle(
            screen, COLOURS["white"], (self.x, self.y), self.radius + 1, 0
        )                                                                                #Draws a circle with pygame by giving screen, circle is white,
                                                                                         #Defines position and size of the circle (increased by 1 in this case) as well
                                                                                         #Keep in mind that this is supposed to be the 'outline' of the circle and not the actual
                                                                                         #target. It it supposed to be make the target standout
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius, 0)         #Defines the actual target circle. the 0 at the end refers to whether the inside
                                                                                         #of the circle will be filled or not.
    def move(self, target_loc, ticks):
        dist_per_tick = self.speed * ticks / 1000                                        #Calculates the distance moved per tick by using operation
                                                                                         #Given as: 'distance = velocity x time'. ticks divided by 1000 to convert seconds into milliseconds
        if (
            abs(self.x - target_loc[0]) <= dist_per_tick                                 #It determines whether the position of the x and y is close enough to the target
            and abs(self.y - target_loc[1]) <= dist_per_tick                             #so that if it keeps going with the same speed, it will pass the target on the next frame
        ):                                                                               #thus when that happens, it is conditioned to 'stop moving'
            self.moving = False
            self.color = COLOURS["red"]                                                  #In that condition, the circle turns red
        else:
            self.moving = True                                                           #Resumes movement of the circle
            self.color = COLOURS["green"]                                                #When moving, the circle turns green
            current_vector = pygame.Vector2((self.x, self.y))                            #Defines the position of the circle as a vector
            new_vector = pygame.Vector2(target_loc)                                      #Defines the target position as another vector
            towards = (new_vector - current_vector).normalize()                          #Defines the speed of the circle per frame   
            self.x += towards[0] * dist_per_tick                                         #Updates position of x based on 'distance = velocity x time' calculation
            self.y += towards[1] * dist_per_tick                                         #Updates position of y based on 'distance = velocity x time' calculation

