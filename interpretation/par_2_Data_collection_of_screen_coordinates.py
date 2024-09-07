

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

                                       #Updates the screen filling with different colours



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


##For calibration mode: defining the 9 screen coordinates of interest
##this returns the shuffle list below:
import itertools

def get_calibration_zones(w, h, target_radius):                                          # w: width of display area, h: height of display area, target_radius: A radius value 
                                                                                         #used to offset the boundary points
    xs = (0 + target_radius, w // 2, w - target_radius)                                  #Defines possible x positions for left most side, centre and right most side
    ys = (0 + target_radius, h // 2, h - target_radius)                                  #Defines possible y positions for left most side, centre and right most side
    zones = list(itertools.product(xs, ys))                                              #Define the zone as all possible above stated as x and y coordinates together
    random.shuffle(zones)                                                                #Shuffle those produced coordinates and keeps the x and y coordinates as pairs
    return zones                                                                         #Returns the shuffled values
w, h = pygame.display.get_surface().get_size()                                           #defines variables w as display surface and h as display size
calibration_zones = get_calibration_zones(w, h, SETTINGS["target_radius"])               #gets target radius value from SETTINGS

##In the main loop: If the calibration mode is chosen
#We draw target at the first location, and when the user
#looks at the target and presses space, the data (xy position and webcam features are saved)

from Target import Target
from pygame.locals import KEYDOWN, K_SPACE

def save_data():
    pygame.image.save(screen, 'screenshot.png')

target = Target(center, speed=SETTINGS["target_speed"], radius=SETTINGS["target_radius"])           #You create this target by input center, speed and size
calibrate_idx = 0                                                                                   #This is set to zero but will be later used to track the current calibration zone  
# In the main loop ...
for event in pygame.event.get():
    if event.type == KEYDOWN and event.key == K_SPACE:                                              #If user presses spacebar and there are more calibration zones left
                                                                                                    #Data is saved (images captured and stored)
                                                                                                    #Calibrate idx incremented to move to next calibration zone
        if calibrate_idx < len(calibration_zones):                                                  #If there are still calibration zones to process
            num_images = save_data(...)                                                             #It saves the image data and moves to next calibration zone
        calibrate_idx += 1
if calibrate_idx < len(calibration_zones):                                                          #If after the loop, there are more calibration zones left
    target.x, target.y = calibration_zones[calibrate_idx]                                           #Match positions to calibration zones
    target.render(screen)                                                                           #Render or 'activate' the target on the screen

##The code that allows you to control calibration in way that the process 
##is not affected by corneal reflections is listed below:

##Below is done at the start of the main-loop

screen.fill(bg)                                                                                     #Fills the screen with black and grey background colour
bg_origin = screen.get_at((0, 0))                                                                   #Gets the current background color in RGBA format at the pixel located top left corner
if bg_origin[0] <= COLOURS["black"][0]:                                                             #bg_origin[0] refers to red component of the screen, if that is less than the red component of
    bg_should_increase = True                                                                       #on black screen, tells screen brightness to increase
elif bg_origin[0] >= COLOURS["gray"][0]:                                                            #Does opposite procedure if red component screen is less than red component of gray
    bg_should_increase = False
if bg_should_increase:
    bg = (bg_origin[0] + 1, bg_origin[1] + 1, bg_origin[2] + 1, bg_origin[3])
else:
    bg = (bg_origin[0] - 1, bg_origin[1] - 1, bg_origin[2] - 1, bg_origin[3])


##Then during data collection mode, target is moved around the screen and
##data is saved at each location:

ticks = clock.tick(SETTINGS["record_frame_rate"])
target.move((x,y), ticks)
target.render(screen)

while True:                                                       #continuously runs the program until stopped manually
    screen.fill(bg)                                                           #Fills entire screen with selected background colour
    l_eye, r_eye, face, face_align, head_pos, angle = detector.get_frame()    #get input frame tensor values from 6 inputs listed below and discussed in previous section
    # ... do things here ...
    ticks = clock.tick(SETTINGS["record_frame_rate"])             #Limits frame rate of the loop to the value specified in SETTINGS dictionary
    pygame.display.update()