

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
import numpy as np


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

ticks = clock.tick(SETTINGS["record_frame_rate"])                           #Stores the number in milliseconds since the previous frame
target.move((x,y), ticks)                                                   #sets the target to move to new position (x,y) with regards to each tick
target.render(screen)                                                       #renders the target on the screen


##Ok but how fast should the target be moving. If too fast, your eyes will have
##very hard time following it when it changes direction, which reduces the validity
##or your collected data. You need to make sure that your eyes are always looking
##at the correct screen location as the target moves. If you move the target too slow,
##eyes will have an easier time to follow but dataset is increased hugely and procees is slowed down
##Speed is determined by values in config.ini
##and correct speed will require experimentation
##Region maps: This is another issue. where does the target rendering start
##and where does it move the most. If we start on target, data sampling would
##disproportionately increase towards the centre and its surroundings
##So the sampling or the number of datasets would form a gaussian distribution originating from the centre
##of the screen producing unequal bias when adjusting parameters.
##To overcome this, the below code was proposed:

region_map = np.zeros((width, height))  # new region map
# When we record from a screen location, we just increment the value in that location of the map:
region_map[x_coord, y_coord] += 1 

##below is the other part of the code which recruits regions wich include
##minimal dataset and returns them into the main loop

def get_undersampled_region(region_map, map_scale):
    min_coords = np.where(region_map == np.min(region_map))                     #defines the minimum coordinates in the region map 
    idx = random.randint(0, len(min_coords[0]) - 1)                             #since there could be multiple regions with minimum value:
                                                                                #idx picks one fo them at random
    return (min_coords[0][idx] * map_scale, min_coords[1][idx] * map_scale)     #calibrate value based on map scale where [0] and [1] refer respectively to
                                                                                #to y and x values
# In main loop....
center = get_undersampled_region(region_map, SETTINGS["map_scale"])             #Sets the centre as one of the minimum regions determined in the function
target.move(center, ticks)
target.render(screen)

#Issues with screen edges:
#region mapping helps for better sampling in terms of coordinates
#that have low data representation. Which is that screen center
#will still have more samples than desired. How do we increase the
#the number of samples at the edges then? a mild and extreme solution has been
#implemented.
    #Extreme case: choose target locations that are at the 4 corners of the screen.
    #Makes it higlhy likely that the target will move along the edges, or diagonally from corner
    #to corner. This is basically moving a version of the calibration mode:

new_x= random.choice([0,w])
new_y=random.choice([0,w])
center=(new_x,new_y)
target.move(center, ticks)
target.render(screen)

#The milder solution would be to increase the probability of sampling
#locations that are near the screen edges. We can use a Beta distribution for this
#BEta distributions come in many forms, but we can choose parameters that result
#in porbability near the boundaries of its range, and low towards the center, 
#which is exactly what we want if we want to prioritize the screen edges

from scipy.stats import beta

new_x = (beta.rvs(0.4, 0.4, size=1) * w)[0]             #Selectively increase the probability of picking coordinate near x edge
new_y = (beta.rvs(0.4, 0.4, size=1) * h)[0]             #Selectively increase the probability of picking coordinate near y edge
center = (new_x, new_y)                                 #incorporates values as variable center
target.move(center,ticks)
target.render(screen)                                   #the code written in previous section written by new code



while True:                                                       #continuously runs the program until stopped manually
    screen.fill(bg)                                                           #Fills entire screen with selected background colour
    l_eye, r_eye, face, face_align, head_pos, angle = detector.get_frame()    #get input frame tensor values from 6 inputs listed below and discussed in previous section
    # ... do things here ...
    ticks = clock.tick(SETTINGS["record_frame_rate"])             #Limits frame rate of the loop to the value specified in SETTINGS dictionary
    pygame.display.update()



