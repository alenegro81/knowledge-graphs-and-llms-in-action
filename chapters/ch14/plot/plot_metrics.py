import matplotlib.pyplot as plt

def plot_metrics(metrics, 
                 metric_key, 
                 title, 
                 xlabel='Epoch', 
                 ylabel=None, 
                 styles=None, 
                 metric_types=None, 
                 figsize=(8, 4)):
    """
    Plots specified metrics for multiple models over epochs.

    Args:
        metrics (dict): A dictionary where keys are model names and values are dictionaries
                        containing dataset metrics (e.g., {'gcn': {'val': {'precisions': [...]}, 'train': {...}}}).
        metric_key (str): The key for the metric to plot (e.g., 'precisions').
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis. Default is 'Epoch'.
        ylabel (str): Label for the y-axis. Default is `metric_key.capitalize()`.
        styles (dict): Optional dictionary for styling. Keys are model names, and values are dictionaries with:
                       {'color': ..., 'linestyle': ..., 'linewidth': ..., 'label': ...}.
        metric_types (list): List of metric types to plot (e.g., ['train', 'val']). Default is `['val']`.
        figsize (tuple): Size of the figure. Default is (8, 4).
    """
    metric_types = metric_types or ['val']
    plt.figure(figsize=figsize)
    
    for model, data in metrics.items():
        for metric_type in metric_types:
            if metric_type not in data:
                continue

            epochs = range(1, len(data[metric_type][metric_key]) + 1)
            style = styles.get(model, {}) if styles else {}
            
            color = style.get('color', None)
            linestyle = style.get('linestyle', '-')
            linewidth = style.get('linewidth', 1.2)
            label = f"{style.get('label', model.upper())} ({metric_type})"
            
            plt.plot(epochs, data[metric_type][metric_key], 
                     color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else metric_key.capitalize())
    plt.title(title)
    plt.legend(fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_multiple_metrics(metrics,
                          metric_keys,
                          titles,
                          xlabel='Epoch',
                          ylabel=None,
                          styles=None,
                          metric_types=None,
                          figsize=(12, 8),
                          cols=2):
    """
    Plots multiple metrics in a grid layout for multiple models over epochs.

    Args:
        metrics (dict): A dictionary where keys are model names and values are dictionaries
                        containing dataset metrics (e.g., {'gcn': {'val': {'precisions': [...]}, 'train': {...}}}).
        metric_keys (list): List of metric keys to plot (e.g., ['accuracies', 'precisions', 'recalls', 'f1_scores']).
        titles (list): Titles for each subplot.
        xlabel (str): Label for the x-axis. Default is 'Epoch'.
        ylabel (str): Label for the y-axis. Default is None (will use `metric_key.capitalize()`).
        styles (dict): Optional dictionary for styling. Keys are model names, and values are dictionaries with:
                       {'color': ..., 'linestyle': ..., 'linewidth': ..., 'label': ...}.
        metric_types (list): List of metric types to plot (e.g., ['train', 'val']). Default is `['val']`.
        figsize (tuple): Size of the entire figure. Default is (12, 8).
        cols (int): Number of columns for the grid layout. Default is 2.
    """
    metric_types = metric_types or ['val']
    num_metrics = len(metric_keys)
    rows = (num_metrics + cols - 1) // cols  # Calculate required rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easy iteration

    for idx, metric_key in enumerate(metric_keys):
        ax = axes[idx]
        for model, data in metrics.items():
            for metric_type in metric_types:
                if metric_type not in data:
                    continue

                epochs = range(1, len(data[metric_type][metric_key]) + 1)
                style = styles.get(model, {}) if styles else {}

                color = style.get('color', None)
                linestyle = style.get('linestyle', '-')
                linewidth = style.get('linewidth', 1.2)
                label = f"{style.get('label', model.upper())} ({metric_type})"

                ax.plot(epochs, data[metric_type][metric_key],
                        color=color, linestyle=linestyle, linewidth=linewidth, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ylabel else metric_key.capitalize())
        ax.set_title(titles[idx])
        ax.legend(fontsize=8)
        ax.grid(False)

    # Hide any unused subplots
    for ax in axes[len(metric_keys):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    metrics = {
        'gcn': {
            'val': {'precisions': [0.7, 0.75, 0.78, 0.82]},
            'train': {'precisions': [0.65, 0.7, 0.73, 0.8]}
        },
        'gat': {
            'val': {'precisions': [0.68, 0.73, 0.77, 0.8]},
            'train': {'precisions': [0.6, 0.65, 0.7, 0.76]}
        },
        'gin': {
            'val': {'precisions': [0.65, 0.7, 0.74, 0.78]},
            'train': {'precisions': [0.62, 0.68, 0.72, 0.75]}
        }
    }

    styles = {
        'gcn': {'color': 'C0', 'linestyle': '--', 'linewidth': 1.2, 'label': 'GCN'},
        'gat': {'color': 'C1', 'linestyle': '-.', 'linewidth': 1.2, 'label': 'GAT'},
        'gin': {'color': 'C2', 'linestyle': ':', 'linewidth': 1.2, 'label': 'GIN'}
    }

    plot_metrics(metrics, 
                    metric_key='precisions', 
                    title='Training and Validation Precisions Across GNN Models',
                    ylabel='Precision', 
                    styles=styles, 
                    metric_types=['train', 'val'])
