import torch
from sklearn.metrics import classification_report, accuracy_score

def evaluate(model, test_loader, device="cuda"):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - device (str): Device to run the model on, default is "cuda".

    Returns:
    - None: Prints accuracy and classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(all_labels, all_preds))
