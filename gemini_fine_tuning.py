#Gemini prediciton
#*******************************#*******************************#*******************************#*******************************#*******************************
#*******************************#*******************************#*******************************#*******************************#*******************************
#*******************************#*******************************#*******************************#*******************************#*******************************
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Dataset Loader
class TemporalEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Gemini Model with Classification Head
class GeminiModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GeminiModel, self).__init__()
        self.hidden_size = hidden_size

        # Transformer-like architecture for temporal embeddings
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=2, dim_feedforward=hidden_size // 2),
            num_layers=2
        )

        # Classification head
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        # Reshape for Transformer (seq_len, batch_size, input_size)
        x = x.permute(1, 0, 2)

        # Pass through Transformer
        transformer_out = self.transformer(x)

        # Use the output of the last time step
        last_hidden_state = transformer_out[-1, :, :]

        # Pass through classification head
        logits = self.fc(last_hidden_state)
        return logits
def load_data(input_file):
    df = pd.read_csv(input_file)
    print(df['Label'].unique())  # Check for unexpected values in the Label column


    # Define a mapping from label strings to integers
    label_mapping = {"Low Risk": 1,"Lowest Risk": 0,"Highest Risk":4, "Medium Risk": 2, "High Risk": 3, "In Default": 5}

    # Map string labels to integers
    df['Label'] = df['Label'].map(label_mapping)

    # Check for missing or invalid labels
    if df['Label'].isnull().any():
        print("Warning: Missing labels detected! Dropping rows with missing labels.")
        df = df.dropna(subset=['Label'])

    # Convert to integer type
    df['Label'] = df['Label'].astype(int)

    # Extract embeddings for each year and stack them
    years = [f"Year_{year}" for year in range(2011, 2017)]
    embeddings = np.array([
        [np.array(eval(row[year]), dtype=np.float32) for year in years] for _, row in df.iterrows()
    ], dtype=np.float32)

    labels = df['Label'].values

    # Debug embeddings and labels
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded labels: {np.unique(labels)}")
    print(f"Labels dtype: {labels.dtype}")

    return embeddings, labels



def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device).float(), labels.to(device)

            # Validate labels
            if (labels < 0).any() or (labels >= model.fc.out_features).any():
                raise ValueError(f"Invalid labels detected in training: {labels}")

            with torch.cuda.amp.autocast():
                outputs = model(embeddings)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

        # Validate on validation set
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device).float(), labels.to(device)
                if (labels < 0).any() or (labels >= model.fc.out_features).any():
                    raise ValueError(f"Invalid labels detected in validation: {labels}")

                with torch.cuda.amp.autocast():
                    outputs = model(embeddings)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.4f}")


# Main script
if __name__ == "__main__":
    input_file = '/content/drive/My Drive/Colab Notebooks/Corporate Credit Risk Prediction/RGCN-master/processed_embeddings_by_subject_predicate.csv'

    # Load data
    embeddings, labels = load_data(input_file)

    # Ensure labels are valid
    valid_indices = (labels >= 0) & (labels < len(np.unique(labels)))
    if not valid_indices.all():
        print(f"Invalid labels found: {labels[~valid_indices]}")
        embeddings = embeddings[valid_indices]
        labels = labels[valid_indices]

    # Check the unique labels
    print(f"Final labels after validation: {np.unique(labels)}")

    print(f"Embeddings Shape: {embeddings.shape}")  # (8294, 6, 300)
    print(f"Labels Shape: {labels.shape}")          # (8294,)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # Create DataLoaders with reduced batch size
    train_dataset = TemporalEmbeddingDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TemporalEmbeddingDataset(torch.tensor(X_val), torch.tensor(y_val))
    # Ensure labels are integers and long tensors
    train_dataset = TemporalEmbeddingDataset(torch.tensor(X_train), torch.tensor(y_train).long())
    val_dataset = TemporalEmbeddingDataset(torch.tensor(X_val), torch.tensor(y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize and train the model
    input_size = embeddings.shape[2]  # Dimension of RGCN embeddings
    hidden_size = 256
    num_classes = len(np.unique(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gemini_model = GeminiModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    train_model(gemini_model, train_loader, val_loader, num_epochs=30, device=device)

    # Evaluate on validation set
    gemini_model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(device).float()
            outputs = gemini_model(embeddings)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())

    print("\nClassification Report:\n", classification_report(val_labels, val_preds))
