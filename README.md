# GenderDetector
This Project was an Semester Project in the order to the Lecture "Special Aspects of mobile Autonomous Systems", where this idea can help the Robots to cummunicate better with people. Whtat did made better is that the whole training process got faster than using normal transfer learning method. Because we found the fine tuning much better to use the high value features plus reduce the time and trainable parameters.\

## Run the code :
First of all let's install some dependencies like , os. Opencv, Python 3.5+, Keras 2.0+, scipy, numpy, Pandas, tqdm, tables, h5py, dlib (for demo)\\
We provided the both models (the one with the keras frozen layers and the solution) , you can as it’s going to be explained just comment one of the model and train this with : python trains.py\

As you can see in the structure we’ve divided the project into this parts :\

Loading the data using data_loader.py class\
Preprocessing and augmenting and normalizing the data using inference.py and preprocessor.py\
Generating the data using generator.py class\
Prepare some dependencies using download.sh and utils\
Models which are located in models folder,where all the needed files like model & .hdf5 & .json format … meant to be saved\
The weights which are located in the folder with the same name\
And the train.py class which is somehow our main for this project\\

If you run the “train.py” , then related model located here: “models/transfer_learning/inception_v3” imported into the class ,will get executed , which is the most recent version of our designed models. By default will the inception_v3 be called.\
By demand you can comment this and uncomment other models, existing in that class. Our previous version of model located here :\
“ models/fine_tuning/inception_v3_finetune”, where we trained the model in two steps, at first on wiki dataset and at second step we freezed some layers and fit the model with imdb dataset, If we want to run code in that mode,then you need to comment the compiling , data_generating , callbacks, checkpoints in the train.py class until visualization part. Because they already defined in that class regarding to the special model architecture that we used.\

As an option we can modify the process by changing this parameters in either argument part or more deep in the code and localy :\
Nb_epochs : number of epochs\ Patience : number of epochs\ that model will continue if val_loss not getting better, while monitoring it \ The optimizer : could be either SGD, Adam, RmsProb \ Batch_size : the number of images per each batch while fitting the model\  Input_shape : should be choose due to architecture of model : 299, 224, 160 \ Validation_split : depends on your decision could be 0.2, 0.1 You can also choose if you monitor loss, val_loss, accuracy, val_acc\

Running after training : How you can run the model on live camera to see the live prediction : \ You should just run that class using this command : Python predictionlivevideo.py \ For running this class we need only to address the weights using “gender_model_path” local parameter .\
