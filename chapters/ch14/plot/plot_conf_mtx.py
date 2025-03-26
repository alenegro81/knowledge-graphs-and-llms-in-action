import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

from ch14.eval.eval_funcs import predict, predict_batched

def generate_confusion_matrices(models, data, mask_type="test"):
    """
    Generates confusion matrices for multiple models on the specified mask.

    Args:
        models (dict): Dictionary where keys are model names and values are trained model instances.
        data (torch_geometric.data.Data): Graph data object.
        mask_type (str): Mask type to use for evaluation ('train', 'test', or 'val').

    Returns:
        dict: Dictionary containing confusion matrices for each model.
    """
    mask_attr = f"{mask_type}_mask"
    if not hasattr(data, mask_attr):
        raise ValueError(f"Invalid mask type: {mask_type}. Valid options are 'train', 'test', or 'val'.")

    mask = getattr(data, mask_attr)
    y_true = data.y[mask].cpu().numpy()

    confusion_matrices = {}

    for model_name, model in models.items():
        # Generate predictions
        test_pred = predict(model, data)[mask]
        y_pred = test_pred.cpu().numpy()

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices[model_name] = cm

    return confusion_matrices

def generate_confusion_matrices_batched(models, data_loader):
    """
    Generates confusion matrices for multiple models using a test DataLoader.

    Args:
        models (dict): Dictionary where keys are model names and values are trained model instances.
        data_loader (torch.utils.data.DataLoader): DataLoader for testing data.

    Returns:
        dict: Dictionary containing confusion matrices for each model.
    """
    confusion_matrices = {}

    for model_name, model in models.items():
        print(f"Generating confusion matrix for {model_name}...")
        
        # Get predictions and ground truth using batched prediction
        preds = []
        ground_truths = []
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch_data in data_loader:
                # Assuming the batch contains the ground truth edge labels
                batch_data = batch_data.to(device)
                ground_truths.append(batch_data["user", "rates", "movie"].edge_label.cpu())
                preds.append(predict_batched(model, [batch_data]))

        # Concatenate predictions and ground truth labels across all batches
        y_pred = torch.cat(preds, dim=0).cpu().numpy()
        y_true = torch.cat(ground_truths, dim=0).cpu().numpy()

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices[model_name] = cm

    return confusion_matrices

def plot_confusion_matrices(confusion_matrices, 
                                     class_names, 
                                     model_names, 
                                     title='Confusion Matrices',
                                     figsize=(12, 8), 
                                     base_color='#0091ea'):
    """
    Plots multiple confusion matrices in one figure without gradient color bars.
    
    Args:
        confusion_matrices (list): List of confusion matrices (numpy arrays).
        class_names (list): List of class labels for axes (e.g., ['illicit', 'licit']).
        model_names (list): List of model names corresponding to each confusion matrix.
        title (str): Title of the plot. Default is 'Confusion Matrices'.
        figsize (tuple): Figure size. Default is (12, 12).
        base_color (str): Hex color code for the gradient. Default is '#0091ea'.
    """
    # Custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue', ['#ffffff', base_color], N=256)

    num_matrices = len(confusion_matrices)
    cols = 2  # Number of columns
    rows = (num_matrices + cols - 1) // cols  # Calculate number of rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easy indexing
    
    for idx, cm in enumerate(confusion_matrices):
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt='g', cmap=custom_cmap,
            annot_kws={'size': 10},
            xticklabels=class_names,
            yticklabels=class_names,
            linecolor='black', linewidth=1,
            square=True, ax=ax
        )
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Actual Class')
        ax.set_title(f'{model_names[idx]}')

        # Annotate with normalized percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm_normalized[i, j] * 100
                text = f'\n({percentage:.1f}%)'
                color = 'white' if percentage > 95 else 'black'
                ax.text(j + 0.5, i + 0.6, text,
                        ha='center', va='center', fontsize=8, color=color)

    for idx in range(len(confusion_matrices), len(axes)):
        fig.delaxes(axes[idx])  # Remove extra subplots

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Example confusion matrices
    cm1 = np.array([[50, 5], [10, 35]])
    cm2 = np.array([[45, 10], [12, 33]])
    cm3 = np.array([[48, 7], [8, 37]])
    cm4 = np.array([[52, 3], [7, 38]])

    # Mapped classes and model names
    mapped_classes = ['illicit', 'licit']
    model_names = ['GCN', 'GAT', 'GIN', 'SGC']

    # Call the function
    plot_confusion_matrices([cm1, cm2, cm3, cm4], 
                                    class_names=mapped_classes, 
                                    model_names=model_names,
                                    title='Confusion Matrices for GNN Models')