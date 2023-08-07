
#NAS PARAMETERS
CONTROLLER_SAMPLING_EPOCHS = 10
SAMPLES_PER_CONTROLLER_EPOCH = 20
CONTROLLER_TRAINING_EPOCHS = 5
ARCHITECTURE_TRAINING_EPOCHS = 5
CONTROLLER_LOSS_ALPHA = 0.9 

#CONTROLLER(LSTM) PARAMETERS
CONTROLLER_LSTM_DIM = 100
CONTROLLER_OPTIMIZER = 'Adam'
CONTROLLER_LEARNING_RATE = 0.01
CONTROLLER_DECAY = 0.1
CONTROLLER_MOMENTUM = 0.0
CONTROLLER_USE_PREDICTOR = False

#CNN ARCH PARAMETERS
MAX_ARCHITECTURE_LENGTH = 8
MLP_DECAY = 0.0
MLP_MOMENTUM = 0.0
MLP_LOSS_FUNCTION = 'categorical_crossentropy'
MLP_ONE_SHOT = True

#As using MNIST dataset for checking of CNN training and testing 
TARGET_CLASSES = 10
TOP_N = 5
