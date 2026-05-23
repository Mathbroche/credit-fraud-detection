import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import FraudDataset

from config import LEARNING_RATE, EPOCHS, BATCH_SIZE, MODEL_PATH

def train(model, X_train, y_train):
    X_train_normal = X_train[y_train == 0]
    dataset = FraudDataset(X_train_normal.values)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    for epoch in range(EPOCHS):
        for X_batch in dataloader :
            optimizer.zero_grad()
            reconstructed = model(X_batch)
            loss = criterion(reconstructed,X_batch )
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        # Affiche toutes les 10 époques
        if (epoch + 1) % 10 == 0:
            print(f"Époque {epoch+1:>3}/{EPOCHS} "
                  f"| Loss : {loss:.6f} ")
            
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modèle sauvegardé → {MODEL_PATH}")
    return losses