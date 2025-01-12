import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Dataset Loader
class TemporalEmbeddingDataset(Dataset):
    def __init__(self, train_embeddings, labels):
        self.train_embeddings = train_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.train_embeddings[idx], self.labels[idx]

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # Classification head

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)  # Initial hidden state
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)  # Initial cell state

        out, (hn, _) = self.lstm(x, (h0, c0))
        logits = self.fc(out[:, -1, :])  # Use the last time step's hidden state
        return logits, hn.squeeze(0)  # Return logits and last-layer embeddings

def load_data(input_file):
    df = pd.read_csv(input_file)
    print("Unique Labels:", df['Label'].unique())

    # Define a mapping from label strings to integers
    label_mapping = {"Low Risk": 0, "Medium Risk": 2, "Highest Risk": 4, "Lowest Risk": 1, "In Default": 5, "High Risk": 3}

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

def train_lstm_and_extract_embeddings(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for train_embeddings, labels in train_loader:
            train_embeddings, labels = train_embeddings.to(device).float(), labels.to(device).long()

            logits, _ = model(train_embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

    # Extract last-layer embeddings from validation set
    model.eval()
    val_embeddings = []
    val_labels = []
    with torch.no_grad():
        for val_embeddings_batch, labels in val_loader:
            val_embeddings_batch = val_embeddings_batch.to(device).float()
            _, last_layer_embeddings = model(val_embeddings_batch)
            val_embeddings.append(last_layer_embeddings.cpu().numpy())
            val_labels.extend(labels.numpy())

    return np.vstack(val_embeddings), np.array(val_labels)

def train_logistic_regression(embeddings, labels):
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

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

    # Split data for training (2011–2015) and validation (2016)
    train_embeddings = embeddings[:, :-1, :]  # Use 2011–2015
    val_embeddings = embeddings[:, -1:, :]  # Use 2016
    train_labels = labels

    # Create DataLoaders
    train_dataset = TemporalEmbeddingDataset(torch.tensor(train_embeddings), torch.tensor(train_labels))
    val_dataset = TemporalEmbeddingDataset(torch.tensor(val_embeddings), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMModel(input_size=train_embeddings.shape[2], hidden_size=128, num_classes=len(np.unique(labels)))
    val_embeddings, val_labels = train_lstm_and_extract_embeddings(lstm_model, train_loader, val_loader, num_epochs=30, device=device)

    # Train Logistic Regression on extracted LSTM embeddings
    train_logistic_regression(val_embeddings, val_labels)
