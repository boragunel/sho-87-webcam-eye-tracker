

#Web stie: https://www.simonho.ca/machine-learning/webcam-eye-tracker-deep-learning-with-pytorch/

#Webcam Eye tracker: Deep learning via PyTorch.
#The problem we have is basically bounding box regression, 
#but simplified to only 2 continuous output values (X-Y screen coordinate)
#Box refers to regions of the image which are marked with a box during object classification
#What the model does is it predicts box's in which the desired object will be
#If the prediction is wrong, it adjust weight parameters and corrects itself
#This is an important example of supoervised learning.
 

import torch
print(torch.__version__)

###Input types are seperated to 5:
#1.Unaligned face (3D image)
#2.Aligned face (3D image)
#3.Right Eye (3D image)
#4.Head position (2D image)
#5.Head angles (Scalar)
###Output types are seperated to 2:
#X and
#Y coordinates

##Goal is to find most accurate model that can map some combination of inputs to
#match ouputs X-Y coordinates, predicting a model of 'best fit'.
#We will be using

#The loss function will depend on 'Mean squared error'. 
#below shows calculation process for MSE:

import math
import numpy as np

list_1=[2,3,4]

def mean_squared_error(n,list,y_predicted):
    size_effect=1/n                      
    list_2=[]
    for iters in list:
        differ=iters-y_predicted
        Mean_sq_differ=math.pow(differ,2)
        list_2.append(Mean_sq_differ)
    sum_of_input_values=sum(list_2)
    Mean_squared_error=sum_of_input_values*size_effect
    print(f'Mean squared error of list 1 = {Mean_squared_error}')
    return Mean_squared_error


#below is an example:

mean_squared_error(len(list_1),list_1,np.mean(list_1))


## We will be using the root of Mean Squared Error
## to represent the real world accuracy of our data:

list_1=[2,3,4]

def Root_of_mean_squared_error(n,list,y_predicted):
    root=math.sqrt(mean_squared_error(n,list,y_predicted))
    print(f'The RMSE for list 1 = {root}')

Root_of_mean_squared_error(len(list_1),list_1,np.mean(list_1))

#Quick revision of how to use OOP's in python
class Car:           
    def __init__(self,make,model,year,color):                     #Initializes a newly created object
        self.make = make
        self.model = model
        self.year = year
        self.color = color                                             #Initializes a new instance (state of'object') of the class
    def drive(self):
        print("This car is driving")
    def stop(self):
        print("this car is stopped")        #These represent methods

car_1=Car('Chevy','Corvet','2021','blue')   #Once you have created the object
car_1.drive()                               #You can use methods within the class on that object
car_1.stop()                                #As the method will only work for input 'self'

print(car_1.make)                           #in the same way you can call for
print(car_1.model)                          #attributes of the object in this way
print(car_1.year)
print(car_1.color)




##The code above symbolizes the 'pixel-wise distance' between our 
##predicted location and the true location.
##for example, an MSE loss value of 10,000 would be equivalent to 100 pixels of inaccuracy in the prediction
##You see in the example above.

###Dataset Overview: Contains 25,738 examples, with 69.01% of screen locations
###being sampled at least once. The entire dataset is 319Mb in size
###We can check how the data is distributed across the screen in reference to a heat map.
###In this case, the coordinates of the screen are defined as X and Y and
###activation or where the participant is focusing is registered as Z axis
###These 3 dimensional vectors describe the positional transformations of the observing eye


###Ingesting data: We first need a way to get our data into our models. For that
#we can use PyTorch Dataset and DataLoader. These allow us to define how data samples are retrieved from disk
#and handles preprocessing, shuffling and batching of the data. The benefit is that we don't need to load the entire dataset
#into memory. Data batches are loaded as needed.

#For the dataset, we can define where the data is stored in the __init__ method. Then the
##__getitem__ method defines what should happen when our DataLoader makes a request
##for data. In this case it simply uses PIL to load the image and applies a few image transformations.

##Reminder for how to use object oriented programming:
#OOP is use to used to put together objects 
#(which have methods and attributes) into a single class making it easily accessible
#Examples:


##Down here shows the class for pulling and handling Face Datasets.
#The class takes the initializes the dataset object
#it then converts the data in such a way that it produces the batch

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

cwd = os.getcwd()

class FaceDataset(Dataset):
    def __init__(self, cwd, data_partial, *img_types):      
        if data_partial:
            self.dir_data = cwd / "data_partial"                #if data is found to be partial, sets the storage directory to 'data_partial'
        else:
            self.dir_data = cwd / "data"                        #In the opposite case, directory is set to 'data'
        df = pd.read_csv(self.dir_data / "positions.csv")       #Read CSV file containing positions. df stands for dataframe    
        df["filename"] = df["id"].astype("str") + ".jpg"        #Create a filename column by appending '.jpg' to 'id'
        self.img_types = list(img_types)                        #sets the image types to the list of acquired image types
        self.filenames = df["filename"].tolist()                        # adminester dataframe filenames into a list 
        self.targets = torch.Tensor(list(zip(df["x"], df["y"])))        # Create a tensor of target positions
        self.head_angle = torch.Tensor(df["head_angle"].tolist())       # Create a tensor of head angles
        self.transform = transforms.Compose(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
             transforms.ToTensor()]                
    )                                                                   #Calibrates the features of the screen to a workable form
    def __len__(self):
        return len(self.targets)                                        #method returns length of target datasets
    def __getitem__(self, idx):             
        batch = {"targets": self.targets[idx]}      # Initialize batch with target positions
        if "head_angle" in self.img_types:          
            batch["head_angle"] = self.head_angle[idx]  # Add head angle to batch if it's in img_types
        for img_type in self.img_types:
            if not img_type == "head_angle":
                img = Image.open(self.dir_data / f"{img_type}" / f"{self.filenames[idx]}")      #Open image
                if img_type == "head_pos":
                    # Head pos is a generated black/white image and shouldn't be augmented
                    img = transforms.ToTensor()(img)        # Convert head_pos image to tensor without augmentation
                else:
                    img = self.transform(img)       # Apply transformations to other images
                batch[img_type] = img                # Add image to batch
        return batch                                # Return the batch


###PyTorch DataLoader:
#The DataLoader handles the task of actually getting a batch of data and passing it to our
#PyTorch models. Here you can control things like the batch size and whether the data
#should be shuffled.
#The below code shows the way this is done, where you want to create the training, validation and test batches


import torch
from torch.utils.data import DataLoader, random_split

def create_datasets(cwd, data_partial, img_types, batch_size=1, train_prop=0.8, val_prop=0.1, seed=87):  #train prop: proportion of dataset to include in training
                                                                                                         #val prop: Proportion of the dataset to include in the validation set
                                                                                                         #seed: Seed for the random number generator to ensure reproducibility. Default is 87.
    
    dataset = FaceDataset(cwd, data_partial, *img_types)                                                 
    n_train = int(len(dataset) * train_prop)
    n_val = int(len(dataset) * val_prop)
    n_test = len(dataset) - n_train - n_val
    ds_train, ds_val, ds_test = random_split(                                                            #Defines the training, validation and test datasets
        dataset, (n_train, n_val, n_test), generator=torch.Generator().manual_seed(seed)                 #shuffle determines whether shuffling data is required at every step
    )                                                                                                    #pin_memory: If this is '= True', data laoder copies Tensors into CUDA pinned memory before returning them to GPU
                                                                                                         #generator allows the random formation of those 3 groups each time (as long as see is kept constant)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True)             #see that now these datasets are converted in a way that it can be passed to pytorch
    return train_loader, val_loader, test_loader


#Face model:
#We'll start by creating a simple model using only the unaligned face image. We can use
#We can use PyTorch lightning for this as it helps to streamline the code and removes a lot of the boilerplate

from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim

class SingleModel(pl.LightningModule):
    def __init__(self, config, img_type):
        super().__init__()                                                      
        self.save_hyperparameters()                                             #This is the starting parameter or 'weight' value before training begins
        featsize=64                                                             #This is the size of the feature map             
        self.example_input_array = torch.rand(1, 3, feat_size, feat_size)       #generates random tensor values between 1 and 3 for the example feature map
        self.img_type = img_type                                                #defines image type as an attribute of the object AI model
        self.lr = config["lr"]                                                  #calls learning rate from config dictionary
        self.filter_size = config["filter_size"]                                #calls back the filter size from the config dictionary
        self.filter_growth = config["filter_growth"]                            #Growth factor for filters (due to nature of convolutional neural 
                                                                                #networks, convolutional filters must grow in size after each layer)
        self.n_filters = config["n_filters"]                                    
        self.n_convs = config["n_convs"]                                        
        self.dense_nodes = config["dense_nodes"]                                # Number of nodes in dense layer 
                                                                                #(in other word for dense layer is fully connected layer, 
                                                                                # it is found nearer towards the end of the process and after all concolutional layers)
        # Defines first convolutional layer
        self.conv_input = nn.Conv2d(3, self.n_filters, self.filter_size)        # nn.Conv2d class takes variables (in_channels, out_channels, kernel_size, ...)
                                                                                #In the referred code it suggests there are '3' input channels (X position, Y position and Head angle)
                                                                                #there are same number of outputs as the number of filters (each convolutional operation by a filter
                                                                                #produce a single output, thus number of filters and outputs are the same)
                                                                                #Finally takes the size of the filter as the size of the kernel
        feat_size = feat_size - (self.filter_size - 1)                          #Represents size of feature map: output size = input size - (filter size -1)
                                                                                #the above shows when there is 0 padding and 1 stride during Convolutional operation

        # Defines next set of convolutional layers via iterating loop
        self.convs1 = nn.ModuleList()                                           #Defines 
        n_out = self.n_filters                                                  #
        for i in range(self.n_convs):                                                            
            n_in = n_out                                                        
            n_out = n_in * self.filter_growth  
            self.convs1.append(self.conv_block(n_in, n_out, self.filter_size))  # Add conv block to list
            feat_size = (feat_size - (self.filter_size - 1)) // 2               # Takes the method shown above for getting feature size, 
                                                                                #passes it to pooling (2 indicates pooling size (2,2) and with stride = 2)
        # FC layers -> output
        self.drop1 = nn.Dropout(0.2)                                            #defines Dropout layer (dropout resets certain number of neurons, in this case 20%,
                                                                                #value to 0 to prevent overfitting and giving inconsistent results)
        self.fc1 = nn.Linear(n_out * feat_size * feat_size, self.dense_nodes)   # First fully connected layer
        self.drop2 = nn.Dropout(0.2)                                            
        self.fc2 = nn.Linear(self.dense_nodes, self.dense_nodes // 2)           #second FC layer
        self.fc3 = nn.Linear(self.dense_nodes // 2, 2)                          # Output layer

    def forward(self, x):                                                       #This basically defines the 'forward pass' of the system
        x = self.conv_input(x)                                                  # Apply first convolutional layer
        for c in self.convs1:                                                   
            x = c(x)                                                            # Apply each conv block
        x = x.reshape(x.shape[0], -1)                                           # Flatten the tensor
        x = self.drop1(F.relu(self.fc1(x)))                                     # Apply first FC layer with ReLU and dropout
        x = self.drop2(F.relu(self.fc2(x)))                                     # Apply second FC layer with ReLU and dropout
        x = self.fc3(x)                                                         # Apply output layer
        return x                                                                # Return output

    def conv_block(self, input_size, output_size, filter_size):                 #defines the convolutional block
        block = nn.Sequential(                                                  #Function takes certain vavlues      
            OrderedDict(                                                        #input, output and convolutional filter size of the Convolutional layer
                [                                                               #It applies the same Relu, normalization and max pooling functions
                    ("conv", nn.Conv2d(input_size, output_size, filter_size)),  
                    ("relu", nn.ReLU()),                                        
                    ("norm", nn.BatchNorm2d(output_size)),                      
                    ("pool", nn.MaxPool2d((2, 2))),                             
                ]
            )
        )
        return block                                                            # Return the conv block
    
    def configure_optimizers(self):                                             #defines optimizer function
        optimizer = optim.Adam(self.parameters(), lr=self.lr)                   
        return optimizer

    def training_step(self, batch, batch_idx):                                                       
        x, y = batch[self.img_type], batch["targets"]           #x refers to activation and y refers to output after putting x into activation function
                                                                # Perform a forward pass through the model to get predictions (y_hat)
        y_hat = self(x)
                                                                # Compute the mean squared error loss (remember that RMSE=pixel-wise errors) between predictions and targets
        loss = F.mse_loss(y_hat, y)
                                                                # Log the training loss for monitoring
        self.log("train_loss", loss)
                                                                
        return loss

    def validation_step(self, batch, batch_idx):
                                                                # Does the exact same thing as the training step but only for validation
        x, y = batch[self.img_type], batch["targets"]                                                        
        y_hat = self(x)                                                        
        val_loss = F.mse_loss(y_hat, y)                                                       
        self.log("val_loss", val_loss)                                                       
        return val_loss

    def test_step(self, batch, batch_idx):                      #Does the exact same thing for training and validation steps
        x, y = batch[self.img_type], batch["targets"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        return loss



from ray import tune
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def train_single(
    config,                                                                                                 #Define configuration settings like a dictionary 
    cwd,
    data_partial,                                                                                           #The fraction of the data that will be used
    img_types,
    num_epochs=1,                                                                                           #Training number over the entire dataset (It remains at but is custom and can be changed when attempting to train)
    num_gpus=-1,                                                                    
    save_checkpoints=False,
):
    d_train, d_val, d_test = create_datasets(cwd, data_partial, img_types, seed=config["seed"], batch_size=config["bs"])        #Uses previous function to create the datasets
    model = SingleModel(config, *img_types)                                                                                     #Initialize model using configuration settings
    trainer = pl.Trainer(                                                                                                       #Sets up ther trainer
        max_epochs=num_epochs,
        gpus=num_gpus,
        accelerator="dp",
        progress_bar_refresh_rate=0,                                                                                            #Disables progress bar
        checkpoint_callback=save_checkpoints,                                                                                   #Decide whether to save checkpoints
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", log_graph=True),                          #Logs training metrics
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")],                                              #Custom callback reporting validation loss
    )
    trainer.fit(model, train_dataloader=d_train, val_dataloaders=d_val)                                                         #Starts the traning process of the model using the training and validation datasets.


import os
os.cwd()

def main():
    train_single(config,cwd,data_partial,img_types,num_epochs=1,num_gpus=-1,save_checkpoints=False)                             #data_partial refers to the proportions of dataset included in
                                                                                                                                #the training, validation and test datasets. you will notice that 
                                                                                                                                #you will notice that data_partial and img_types will be initially undefiend
if __name__ == '__main__':                                                                                                      #That is because you need the dataset to be linked to the current working directory
    main()



###Ray Tune: we need to wrap the training function in some Ray Tune.
###Ray Tune is provided in the form of a python library as including methods
### for tuning hyperparameters and runnning experiments
##There is a nice example in the following:

from ray import train, tune

from ray import train, tune


def objective(config):  # ①
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


search_space = {  # ②
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),                 #tune.grid_search tries every choice possible
    "b": tune.choice([1, 2, 3]),                                    #tune.choice() picks a number on random and tries it
}

tuner = tune.Tuner(objective, param_space=search_space)             #Tests possible combinations for a and b

results = tuner.fit()                                               #returns the values from parameter testing
print(results.get_best_result(metric="score", mode="min").config)   #Prints results


##You see that for the case above, you wanted to get the minimum values
##which was accomplished by ray tuning.The idea is to try multiple values
#and combinations to find the models of best fit. 

import datetime
from pathlib import Path

from ray import tune
from ray.tune.schedulers import ASHAScheduler

#Traditionally, when you tune hyperparameters using grid search or 
# random search, you fully train all of your model/hyperparameters 
# combinations. This can be a waste of resources, because you can tell early on that 
# some models just won’t work well. ASHA is a halving algorithm that prunes poor performing models, 
# and only fully train the best models.
#the function below indeed tries out multiple models and 'tune' by only selecting
#the ones which show the minimal loss value.

def tune_asha(
    config,                                                                                                
    train_func,
    name,
    img_types,
    num_samples,
    num_epochs,
    data_partial=False,
    save_checkpoints=False,
    seed=1,
):
    cwd = Path.cwd()                                                                        #chooses current working directory to store the logs
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)         #max_t: max epochs per trial
                                                                                            #grace_period: minimum epochs that should be run per trial
                                                                                            #reduction factor: how the trials are stoppped
    reporter = tune.CLIReporter(                                                            #Define CLI to show progress of tuning in the terminal
        parameter_columns=list(config.keys()),                                              #shows the tuned hyperparameters
        metric_columns=["loss", "training_iteration"],)                                     #converted to CLI reported as JupyterNotebook reporter not essential
                                                                                            #the metric_column show the loss value per training of the model
    
    analysis = tune.run(                                                                    #runs the tuning
        tune.with_parameters(                                                               #recalls the training functions to run the model
            train_func,                                                                     #with parameter values given at the start of the function
            cwd=cwd,
            data_partial=data_partial,
            img_types=img_types,
            save_checkpoints=save_checkpoints,
            num_epochs=num_epochs,
            num_gpus=1,
        ),
        resources_per_trial={"cpu": 2, "gpu": 1},
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="{}/{}".format(
            name, datetime.datetime.now().strftime("%Y-%b-%d %H-%M-%S") #tells the program to set a name for running the tuning session
        ),
        local_dir=cwd / "logs"                                          #stores the logs within a folder of the cwd named logs
    )



###Training unaligned face model: With all the helper functions defined, training the 
# model is as simple as providing a range of hyperparameters values as a config 
# dictrionary, and calling our tune function. PyTorch also allows us to log training
# results to Tensorboard for analysis
#We start by exploring a wide range of values to get a sense of what the search space looks like:

config = {
        "seed": tune.randint(0, 10000),  # reproducible random seed for each experiment
        "bs": tune.choice([1 << i for i in range(2, 9)]),  # batch size
        "lr": tune.loguniform(1e-7, 1e-3),  # learning rate
        "filter_size": tune.choice([3, 5, 7, 9]),  # filter size (square)
        "filter_growth": tune.choice([1, 2]),  # increase filter count by a factor
        "n_filters": tune.choice([8, 16, 32, 64]),  # number of starting filters
        "n_convs": tune.choice([0, 1, 2]),  # number of conv layers
        "dense_nodes": tune.choice([16, 32, 64, 128]),  # number of nodes in fc layer
    }
analysis = tune_asha(config, data_partial=True, train_func=train_single, name="face/explore", img_types=["face"], num_samples=100, num_epochs=10, seed=87)

###The above variable named as analysis is for running the hyperparameter tuning based on
###the defined values given so far within the script.

###If we actually train and use validation for the graphs, and model (see image on the website):
###we actually see that ASHA prunes poorly performing models to save time.
###So you basically have constructed a system which iterates through multiple different models,
###and spits out the model which fits the patterns of a training and validation curve.

###If you see the picture from the other graph, you can see that the 
##They have smaller batch sizes, learning rate is around (1x10^-4)
##and a larger number of fully connected (dense) nodes.

###hyperparameters ranges and search again over more epochs.
###After a second round of search, we take the best performing
###hyperparameters and train the final model over 50 epochs.

import json
from pathlib import Path
from utils import get_best_results

start_time = datetime.datetime.now().strftime("%Y-%b-%d %H-%M-%S")                                                                           #captures current date/time and format as string
config = get_best_results(Path.cwd()/"logs"/"face"/"tune")                                                                                   #gets the best hyperparameter values
pl.seed_everything(config["seed"])                                                                                                           #Determines the seed which will be constant
d_train, d_val, d_test = create_datasets(Path.cwd(), data_partial=True, img_types=["face"], seed=config["seed"], batch_size=config["bs"])    #Creates the dataset and set variables for each class
model = SingleModel(config, "face")                                                                                                          #Define first model as object
trainer = pl.Trainer(
    max_epochs=50,                                                                                                                           #sets epoch number
    gpus=[0, 1],
    accelerator="dp",                                                                                                                        #sets the GPU pathway
    checkpoint_callback=True,                                                                                                                #ensures model checkpoints will be saved
    logger=TensorBoardLogger(save_dir=Path.cwd()/"logs", name="face/final/{}".format(start_time), log_graph=True))                           #Log training metrics
trainer.fit(model, train_dataloader=d_train, val_dataloaders=d_val)                                                                          #This starts the trianing process
test_results = trainer.test(test_dataloaders=d_test)                                                                                         #gets back the results from the test

###On the test set we get an MSE loss of 2362, which is a pixel error of 48.6 pixels.
###the face model fed with aligned faces give an MSE loss of 2539 or 50.4 pixels.
###Performance with aligned faces is a bit worse, its possible that head angle is an important feature for eye tracking
###and is being learned indirectly from the unaligned face image through multiple convolutions.

###For an eye tracker, it is important that a model is fed images which only contain eyes
#Complicated: as there are two input images, need to add a second network of convolutions
#and merge results from the left and right eye image.
###After completing the fine tuning for the model, you find that MSE is 61.9 pixels
##which is significantly worse than the previous two models
##Likely due convolutions on face images easily identifying eyes section and filtering it out

###Summary: 
#   Unaligned face: 48.6 pixel error
#   Aligned face: 50.4 pixel error
#   Eyes: 61.9 pixel error



