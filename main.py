from src.data import load_data, preprocess, split_data
from src.model import Autoencoder
from src.train import train
from src.evaluate import evaluate
from config import DATA_PATH, INPUT_DIM, HIDDEN_DIM, ENCODING_DIM
import matplotlib.pyplot as plt

df = load_data(DATA_PATH)

X,y,scaler = preprocess(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y)
model = Autoencoder(INPUT_DIM, HIDDEN_DIM, ENCODING_DIM)
losses = train(model, X_train, y_train)
evaluate(model, X_train, y_train, X_test, y_test)

plt.plot(losses)
plt.title("Loss pendant l'entraînement")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()