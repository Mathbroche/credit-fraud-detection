DATA_PATH = "data/creditcard.csv"

INPUT_DIM = 30          # V1-V28 + Amount + Time (sans Class) (donc 28+2)
ENCODING_DIM = 15    # dimension du latent space 
HIDDEN_DIM = 64      # dimension des couches intermédiaires

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 256

THRESHOLD_PERCENTILE = 99  

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

MODEL_PATH = "autoencoder.pth"