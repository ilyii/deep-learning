import spacy

# Settings
nlp = spacy.load("en_core_web_sm")
data_path = "data/"
device = "cuda"

# k for the k most occurring words
k = 1000
# Key to decide if we want to use cleaned words or not
wkey = "clean_words"  # "words" or "clean_words"

# Single data point settings (Single Line)
SL_EMBEDDING_DIM = 128
SL_HIDDEN_DIM = 256
SL_EPOCHS = 3
SL_LR = 0.005
SL_DROPOUT = 0.0

# Batch data point settings (Batch Line)
BL_EMBEDDING_DIM = 768
BL_HIDDEN_DIM = 768
BL_EPOCHS = 10
BL_LR = 0.0004
BL_LAYER = 2  # Layer of the LSTM, GRU
BL_DROPOUT = 0.3
BATCH_SIZE = 64
