import torch
import time

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

def initialize_metrics_storage():
    return {
        'losses': [],
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': []
    }

def train_step(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate_step(model, data):
    return calculate_metrics(model, data, 'val')

def train(num_epochs, data, model, optimizer, criterion):
    # Initialize metrics storage
    train_metrics = initialize_metrics_storage()
    val_metrics = initialize_metrics_storage()

    for epoch in range(1, num_epochs + 1):
        # Training Step
        train_loss = train_step(model, optimizer, criterion, data)
        train_metrics_epoch = calculate_metrics(model, data, 'train')
        update_metrics(train_metrics, train_metrics_epoch, train_loss)

        # Validation Step
        val_metrics_epoch = validate_step(model, data)
        update_metrics(val_metrics, val_metrics_epoch)

        # Logging
        if epoch % 100 == 0:
            log_epoch(epoch, train_loss, train_metrics_epoch, val_metrics_epoch)

    return {
        'train': train_metrics,
        'val': val_metrics
    }

def calculate_metrics(model, data, mask_type='train'):
    mask = getattr(data, f"{mask_type}_mask")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        correct = (pred == data.y[mask]).sum()
        accuracy = int(correct) / int(mask.sum())

        y_true = data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()

        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def update_metrics(metrics, metrics_epoch, loss=None):
    if loss is not None:
        metrics['losses'].append(loss)
    metrics['accuracies'].append(metrics_epoch['accuracy'])
    metrics['precisions'].append(metrics_epoch['precision'])
    metrics['recalls'].append(metrics_epoch['recall'])
    metrics['f1_scores'].append(metrics_epoch['f1_score'])

def log_epoch(epoch, train_loss, train_metrics_epoch, val_metrics_epoch):
    print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Train - '
          f'Acc: {train_metrics_epoch["accuracy"]:.4f} - '
          f'Prec: {train_metrics_epoch["precision"]:.4f} - '
          f'Rec: {train_metrics_epoch["recall"]:.4f} - '
          f'F1: {train_metrics_epoch["f1_score"]:.4f}')
    print(f'Val - Acc: {val_metrics_epoch["accuracy"]:.4f} - '
          f'Prec: {val_metrics_epoch["precision"]:.4f} - '
          f'Rec: {val_metrics_epoch["recall"]:.4f} - '
          f'F1: {val_metrics_epoch["f1_score"]:.4f}')

def train_multi_models(classifier,
                       models, 
                       data,
                       hidden_dim,
                       num_classes,
                       num_epochs=100,
                       lr=0.01,
                       weight_decay=0.0005, 
                       device='cuda'):
    """
    Trains and evaluates multiple models for the classification task

    Args:
        classifier (torch.nn.Module): Classifier model.
        models (dict): Dictionary where keys are model names and values are model classes (uninstantiated).
        data (torch_geometric.data.Data): Graph data object.
        hidden_dim (int): Hidden dimension for the model.
        num_classes (int): Number of target classes.
        num_epochs (int): Number of epochs for training. Default is 400.
        lr (float): Learning rate. Default is 0.01.
        weight_decay (float): Weight decay for the optimizer. Default is 0.0005.
        device (str): Device to run the models on ('cuda' or 'cpu').

    Returns:
        dict: Dictionary containing training and validation metrics for all models.
        dict: Dictionary containing the trained model instances for all models.
    """
    # Prepare data and loss criterion
    data = data.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    metrics = {}
    trained_models = {}  # To store the trained model instances

    for model_name, model_class in models.items():
        print(f"\n### Training {model_name}...")

        # Instantiate the model and move it to the device
        model = classifier(model_class(input_dim=data.num_features, 
                                       hidden_dim=hidden_dim,
                                       out_dim=num_classes)).to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Record the start time
        start_time = time.time()

        # Train the model
        train_val_metrics = train(num_epochs, data, model, optimizer, criterion)

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Update the global metrics dictionary
        metrics[model_name] = train_val_metrics

        # Store the trained model
        trained_models[model_name] = model

        print(f"{model_name} training completed in {elapsed_time:.2f} seconds.")

    return metrics, trained_models
