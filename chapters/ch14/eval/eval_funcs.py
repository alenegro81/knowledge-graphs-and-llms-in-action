import torch

def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    return pred

def predict_batched(model, data_loader):
    """
    Predicts labels for batched data using a DataLoader.

    Args:
        model (torch.nn.Module): Trained model to use for predictions.
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of graph data.

    Returns:
        torch.Tensor: Predicted labels for all batches.
    """
    model.eval()
    preds = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)
            probs = torch.sigmoid(out)
            preds.append((probs >= 0.5).float())

    return torch.cat(preds, dim=0)

def predict_probabilities(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probabilities = torch.exp(out)
    return probabilities

def predict_probabilities_batched(model, dataloader, device='cpu'):
    model.eval()
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = batch.to(device)
            # Extract node features and edge indices for the specified edge type
            # x = batch[edge_type[0]].x
            # edge_index = batch[edge_type].edge_index

            # Forward pass
            # out = model(x, edge_index)
            out = model(batch)

            # If output is logits, apply sigmoid to get probabilities
            probabilities = torch.sigmoid(out)

            # Collect probabilities
            all_probabilities.append(probabilities)

    # Concatenate all probabilities into a single tensor
    return torch.cat(all_probabilities, dim=0)