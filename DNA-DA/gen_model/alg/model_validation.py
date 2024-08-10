import torch
from sklearn.metrics import confusion_matrix


def get_accuracy_user(network, dataloader, num_classes):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    network.eval()
    with torch.no_grad():
        for data in dataloader:
            x = data[0].float()
            y = data[1].long()

            p = network.predict(x)

            correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(y)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(p.argmax(1).cpu().numpy())

    network.train()

    accuracy = correct / total
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return accuracy, conf_matrix

# Example usage:
# accuracy, conf_matrix = get_accuracy_user(network, dataloader, num_classes=10)
# print(f'Accuracy: {accuracy}')
# print(f'Confusion Matrix:\n{conf_matrix}')
