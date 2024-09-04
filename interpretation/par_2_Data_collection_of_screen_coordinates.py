

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






