from sklearn.metrics import classification_report
from ch14.eval.eval_funcs import predict

def show_classification_reports(model_name, data, train_pred, test_pred, mapped_classes):
    """
    Prints the classification report for training and testing data.

    Args:
        data (torch_geometric.data.Data): Graph data object containing masks.
        train_pred (torch.Tensor): Predictions for the training data.
        test_pred (torch.Tensor): Predictions for the testing data.
        mapped_classes (list): List of class names mapped to target indices.
    """
    print("\n===============================")
    print(f"Classification Report {model_name}")
    print("===============================\n")

    # Train
    y_true_train = data.y[data.train_mask].cpu().numpy()
    y_pred_train = train_pred.cpu().numpy()

    report_train = classification_report(y_true_train, y_pred_train, target_names=mapped_classes)

    print(f"{4*' '}TRAIN")
    print("---------")
    print(report_train)

    # Test
    y_true_test = data.y[data.test_mask].cpu().numpy()
    y_pred_test = test_pred.cpu().numpy()

    report_test = classification_report(y_true_test, y_pred_test, target_names=mapped_classes)

    print(f"{4*' '}TEST")
    print("--------")
    print(report_test)
  
def show_multiple_reports(models, data, mapped_classes):
  for model_name, model in models.items():
      train_pred = predict(model, data)[data.train_mask]
      test_pred = predict(model, data)[data.test_mask]
      show_classification_reports(model_name, data, train_pred, test_pred, mapped_classes)
