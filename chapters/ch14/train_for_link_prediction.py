import tqdm
import torch
import time
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def initialize_metrics_storage():
    return {
        'losses': [],
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': []
    }

def train_step(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    total_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
    }

    for batch_data in tqdm.tqdm(train_loader, desc="Training Batches"):
        optimizer.zero_grad()
        batch_data.to(device)
        out = model(batch_data)
        ground = batch_data["user", "rates", "movie"].edge_label.to(device)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(out, ground)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Compute metrics
        probs = torch.sigmoid(out)
        preds = (probs >= 0.5).float()
        y_true = ground.cpu().numpy()
        y_pred = preds.cpu().numpy()
        total_metrics['accuracy'] += accuracy_score(y_true, y_pred)
        total_metrics['precision'] += precision_score(y_true, y_pred, average='weighted', zero_division=0)
        total_metrics['recall'] += recall_score(y_true, y_pred, average='weighted', zero_division=0)
        total_metrics['f1_score'] += f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Average loss and metrics across batches
    avg_loss = total_loss / len(train_loader)
    for key in total_metrics:
        total_metrics[key] /= len(train_loader)

    return avg_loss, total_metrics

def validate_step(model, val_loader, device):
    model.eval()
    total_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
    }

    with torch.no_grad():
        for batch_data in tqdm.tqdm(val_loader, desc="Validation Batches"):
            batch_data.to(device)
            out = model(batch_data)
            ground = batch_data["user", "rates", "movie"].edge_label.to(device)
            probs = torch.sigmoid(out)
            preds = (probs >= 0.5).float()

            # Metrics calculation
            y_true = ground.cpu().numpy()
            y_pred = preds.cpu().numpy()
            total_metrics['accuracy'] += accuracy_score(y_true, y_pred)
            total_metrics['precision'] += precision_score(y_true, y_pred, average='weighted', zero_division=0)
            total_metrics['recall'] += recall_score(y_true, y_pred, average='weighted', zero_division=0)
            total_metrics['f1_score'] += f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Average metrics across batches
    for key in total_metrics:
        total_metrics[key] /= len(val_loader)

    return total_metrics

def train(num_epochs, train_loader, val_loader, model, optimizer, device):
    train_metrics = initialize_metrics_storage()
    val_metrics = initialize_metrics_storage()

    for epoch in range(1, num_epochs + 1):
        # Training Step
        train_loss, train_metrics_epoch = train_step(model, optimizer, train_loader, device)
        update_metrics(train_metrics, train_metrics_epoch, train_loss)

        # Validation Step
        val_metrics_epoch = validate_step(model, val_loader, device)
        update_metrics(val_metrics, val_metrics_epoch)

        # Logging
        log_epoch(epoch, train_loss, train_metrics_epoch, val_metrics_epoch)

    return {
        'train': train_metrics,
        'val': val_metrics
    }


def update_metrics(metrics, metrics_epoch, loss=None):
    if loss is not None:
        metrics['losses'].append(loss)
    metrics['accuracies'].append(metrics_epoch.get('accuracy', 0))
    metrics['precisions'].append(metrics_epoch.get('precision', 0))
    metrics['recalls'].append(metrics_epoch.get('recall', 0))
    metrics['f1_scores'].append(metrics_epoch.get('f1_score', 0))


def log_epoch(epoch, train_loss, train_metrics_epoch, val_metrics_epoch):
    print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, "
          f"Train - Acc: {train_metrics_epoch['accuracy']:.4f}, "
          f"Prec: {train_metrics_epoch['precision']:.4f}, "
          f"Rec: {train_metrics_epoch['recall']:.4f}, "
          f"F1: {train_metrics_epoch['f1_score']:.4f}")
    print(f"Validation - Acc: {val_metrics_epoch['accuracy']:.4f}, "
          f"Prec: {val_metrics_epoch['precision']:.4f}, "
          f"Rec: {val_metrics_epoch['recall']:.4f}, "
          f"F1: {val_metrics_epoch['f1_score']:.4f}")


def train_multi_models(classifier, models, data, train_loader, val_loader, test_loader=None,
                       hidden_dim=64, out_dim=1, num_epochs=100, lr=0.01, weight_decay=0.0005, device='cuda'):
    """
    Trains multiple GNN models with a given classifier and returns metrics and trained models.

    Parameters:
        classifier: The classifier class to use (e.g., MovieLensLinkPredictor).
        models (dict): A dictionary of model names and their corresponding classes (e.g., GAT, GCN, SAGE).
        data: The dataset containing graph information and features.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader (optional): DataLoader for testing.
        hidden_dim (int): Hidden dimension size.
        out_dim (int): Output dimension size.
        num_epochs (int): Number of epochs for training.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        device (str): Device to use for training ('cuda' or 'cpu').

    Returns:
        dict: A dictionary of metrics for each model.
        dict: A dictionary of trained model instances for each model.
    """
    metrics = {}
    trained_models = {}

    for model_name, model_class in models.items():
        print(f"\n### Training {model_name}...")

        # Instantiate the model using the classifier
        model = classifier(
            gnn_model=model_class,  # The GNN model (e.g., GAT, GCN, SAGE)
            data=data,              # Full heterogeneous data (to set the embedding dimension)
            hidden_channels=hidden_dim  # Hidden dimension size
        ).to(device)

        # Record the start time
        start_time = time.time()

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train the model
        train_val_metrics = train(num_epochs, train_loader, val_loader, model, optimizer, device)

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time

        metrics[model_name] = train_val_metrics
        trained_models[model_name] = model

        print(f"{model_name} training completed in {elapsed_time:.2f} seconds.")

    return metrics, trained_models