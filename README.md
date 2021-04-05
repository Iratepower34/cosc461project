# cosc461project
## Project for COSC 461 Winter 2021
This is a program that uses a convolutional neural net to try and solve handwritten equations.
### How to run
This program runs in Python 3 and uses
* OpenCV
* Tensorflow2
In order to train a model, you need to create a dataset. Edit `create_training_set.py` in order to do so.
A model is already included in the repo, but if you make alterations to the dataset or model itself you can run `train_model.py` to retrain.
Finally, running `image_processing.py` to test the model and the program (hopefully) prints a prediction to console.
### Notes
This is my first attempt at even using tensorflow, much less fully understanding how to develop a CNN.
### Known limitations
This program is designed to only recognizes digits 0-9 and the 4 basic operators. Anything else isn't included.
