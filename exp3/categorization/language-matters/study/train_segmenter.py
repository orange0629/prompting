import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Complete the load_segment_dataset function
def load_segment_dataset():
    import glob
    import json
    data = []
    for jsonl_filename in glob.glob("*o3_mini_result.jsonl"):
        with open(jsonl_filename, 'r') as f:
            for line in f:
                payload = json.loads(line)
                labels = []
                prev = None
                for segment in payload['output'].split('<sep>'):
                    if prev is not None:
                        curr = [s for s in segment.split('\n') if len(s) ][0]
                        labels.append(( [prev, curr], 1))
                        prev = None

                    for chunk in segment.split('\n'):
                        if len(chunk) == 0:
                            continue
                        if prev:
                            labels.append(( [prev, chunk], 0))
                        prev = chunk
                data += labels
    for jsonl_filename in glob.glob("*gpt-4o-2024-11-20.jsonl"):
        with open(jsonl_filename, 'r') as f:
            for line in f:
                payload = json.loads(line)
                labels = []
                prev = None
                for segment in payload['output'].split('<sep>'):
                    if prev is not None:
                        curr = [s for s in segment.split('\n') if len(s) ][0]
                        labels.append(( [prev, curr], 1))
                        prev = None

                    for chunk in segment.split('\n'):
                        if len(chunk) == 0:
                            continue
                        if prev:
                            labels.append(( [prev, chunk], 0))
                        prev = chunk
                data += labels

    return data

# Dataset class for sentence pairs
class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, labels):
        """
        Dataset for sentence pairs and their labels
        
        Args:
            sentence_pairs (list): List of sentence pairs
            labels (list): List of labels (0 or 1)
        """
        self.sentence_pairs = sentence_pairs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sentence_pairs[idx], self.labels[idx]

# Custom batch encoder to encode sentences in batches
class BatchEncoder:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        """
        Encodes batches of sentence pairs using a SentenceTransformer model
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def __call__(self, batch):
        """
        Encode a batch of sentence pairs
        
        Args:
            batch (list): List of (sentence_pair, label) tuples
        
        Returns:
            tuple: (embeddings1, embeddings2, labels) tensors
        """
        sentence_pairs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Split pairs into two lists
        sentence1 = [pair[0] for pair in sentence_pairs]
        sentence2 = [pair[1] for pair in sentence_pairs]
        
        # Encode sentences
        embeddings1 = self.model.encode(sentence1, convert_to_tensor=True, normalize_embeddings=True)
        embeddings2 = self.model.encode(sentence2, convert_to_tensor=True, normalize_embeddings=True)
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return embeddings1, embeddings2, labels

# Classifier model for sentence pair classification
class SentencePairClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        """
        Classifier for sentence pair embeddings that determines if they're different steps
        
        Args:
            embedding_dim (int): Dimension of sentence embeddings
            hidden_dim (int): Dimension of hidden layer
        """
        super(SentencePairClassifier, self).__init__()
        # Input: concatenated embeddings and cosine similarity
        self.fc1 = nn.Linear(embedding_dim * 2 + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, embeddings1, embeddings2):
        """
        Forward pass
        
        Args:
            embeddings1 (torch.Tensor): Embeddings of first sentences
            embeddings2 (torch.Tensor): Embeddings of second sentences
        
        Returns:
            torch.Tensor: Raw logits (apply sigmoid for probabilities)
        """
        # Compute cosine similarity between embeddings
        cosine_sim = torch.sum(embeddings1 * embeddings2, dim=1, keepdim=True)
        
        # Concatenate embeddings and similarity
        combined = torch.cat((embeddings1, embeddings2, cosine_sim), dim=1)
        
        # Pass through layers
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Training function for the classifier
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    """
    Train the classifier model
    
    Args:
        model (SentencePairClassifier): Classifier model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of epochs to train for
        lr (float): Learning rate
        device (str): Device to train on ('cuda' or 'cpu')
    
    Returns:
        SentencePairClassifier: Trained model
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for embeddings1, embeddings2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            embeddings1 = embeddings1.to(device)
            embeddings2 = embeddings2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings1, embeddings2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_true_pos = 0
        val_false_pos = 0
        val_false_neg = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for embeddings1, embeddings2, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                embeddings1 = embeddings1.to(device)
                embeddings2 = embeddings2.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings1, embeddings2).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                # Metrics for accuracy
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                # Metrics for F1 score
                val_true_pos += ((predictions == 1) & (labels == 1)).sum().item()
                val_false_pos += ((predictions == 1) & (labels == 0)).sum().item()
                val_false_neg += ((predictions == 0) & (labels == 1)).sum().item()
                
                # Store predictions and true labels for further analysis if needed
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Calculate precision, recall, and F1 score
        val_precision = val_true_pos / (val_true_pos + val_false_pos) if (val_true_pos + val_false_pos) > 0 else 0
        val_recall = val_true_pos / (val_true_pos + val_false_neg) if (val_true_pos + val_false_neg) > 0 else 0
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Save the best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_f1_sentence_pair_classifier.pt')
            print(f"Saved new best model with F1: {val_f1:.4f}")
            
        # Also save based on accuracy for comparison
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_acc_sentence_pair_classifier.pt')
            print(f"Saved new best model with Accuracy: {val_acc:.4f}")
    
    return model

# Main function to run the training pipeline
def main():
    """
    Main function to load data, train model, and save it
    """
    # Load dataset
    print("Loading dataset...")
    data = load_segment_dataset()
    
    # Print class distribution
    labels = [item[1] for item in data]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    print(f"Class distribution:")
    print(f"Class 1 (Different steps): {positive_count} ({positive_count/len(labels)*100:.2f}%)")
    print(f"Class 0 (Same step): {negative_count} ({negative_count/len(labels)*100:.2f}%)")
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=labels)
    
    # Extract sentence pairs and labels
    train_pairs = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]
    val_pairs = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]
    
    # Create datasets
    train_dataset = SentencePairDataset(train_pairs, train_labels)
    val_dataset = SentencePairDataset(val_pairs, val_labels)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create batch encoder
    print("Creating batch encoder with SentenceTransformer model...")
    batch_encoder = BatchEncoder()
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=batch_encoder)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_encoder)
    
    # Get embedding dimension from the model
    model = SentenceTransformer("intfloat/multilingual-e5-base", trust_remote_code=True)
    sample_embedding = model.encode("样例文本", convert_to_tensor=True)
    embedding_dim = sample_embedding.size(0)
    print(f"Embedding dimension: {embedding_dim}")
    
    # Create classifier
    print("Creating classifier model...")
    classifier = SentencePairClassifier(embedding_dim, hidden_dim=512)
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    classifier = train_model(
        classifier,
        train_loader,
        val_loader,
        epochs=30,
        lr=5e-4,
        device=device
    )
    
    # Load best F1 model for return
    best_model_path = 'best_f1_sentence_pair_classifier.pt'
    classifier.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")
    
    return classifier, batch_encoder.model

# Function for making predictions on new sentence pairs
def predict(classifier, sentence_model, sentence1, sentence2, device='cuda'):
    """
    Make a prediction for a pair of sentences
    
    Args:
        classifier (SentencePairClassifier): Trained classifier model
        sentence_model (SentenceTransformer): SentenceTransformer model for encoding
        sentence1 (str): First sentence
        sentence2 (str): Second sentence
        device (str): Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        tuple: (probability, predicted_label)
    """
    classifier = classifier.to(device)
    classifier.eval()
    
    # Encode sentences
    embedding1 = sentence_model.encode(sentence1, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0).to(device)
    embedding2 = sentence_model.encode(sentence2, convert_to_tensor=True, normalize_embeddings=True).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = classifier(embedding1, embedding2)
        prob = torch.sigmoid(output).item()
    
    # Return probability and predicted label
    label = 1 if prob > 0.5 else 0
    return prob, label

# Example usage
if __name__ == "__main__":
    # Train the model
    classifier, sentence_model = main()
    
    # Example prediction
    sentence1 = "样例数据-1"
    sentence2 = "样例数据-2"
    prob, label = predict(classifier, sentence_model, sentence1, sentence2)
    
    print(f"Probability: {prob:.4f}, Predicted Label: {label}")
    print(f"Sentences '{sentence1}' and '{sentence2}' are {'different steps' if label == 1 else 'the same step'} in a process")