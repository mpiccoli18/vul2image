import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import time
import os
import numpy as np
from PIL import Image
from torchvision import transforms


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
TrainFilePath = 'output/dataset_output'
num_epochs = 100
train_batch_size = 256
eval_batch_size = 512
learning_rate = 0.0003
dropout1 = 0
dropout2 = 0
dropout3 = 0

# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert to a tensor and automatically normalize to [0, 1]
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize if needed
# ])


average_ACC = 0
average_TPR = 0
average_TNR = 0
average_Precision = 0

# Define the model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout1)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.dropout2 = nn.Dropout(dropout2)

        self.fc1 = nn.Linear(64 * 13 * 13, 81)  # Adjust according to the image dimensions
        self.dropout3 = nn.Dropout(dropout3)
        self.fc2 = nn.Linear(81, 2)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = self.pool(x)
        x = self.dropout1(x)


        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = self.pool(x)
        x = self.dropout2(x)


        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        #return torch.sigmoid(x)
        return x


# Train and evaluate the model
def train_model(model, X_train, Y_train, batch_size, epochs):
    # Data loader
    train_data = TensorDataset(X_train, Y_train)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            #inputs = inputs.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


# Confusion matrix
def evaluate_model(model, X_test, Y_true, eval_batch_size, test_names=None):
    global average_ACC, average_TPR, average_TNR, average_Precision, cost_time
    model.eval()
    model.to(device)

    test_data = TensorDataset(X_test, Y_true)
    test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False)

    total_accuracy = 0
    total_tpr = 0
    total_tnr = 0
    total_precision = 0
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_f1 = 0

    sample_offset = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(inputs)):
                nonVul = probs[i, 0].item() * 100.0
                vul = probs[i, 1].item() * 100.0
                trueLabel = int(labels[i].item())
                predLabel = int(predicted[i].item())

                if test_names is not None and sample_offset + i < len(test_names):
                    img_label = test_names[sample_offset + i]
                else:
                    img_label = f"sample_{sample_offset + i}"

                print(
                    f"[{img_label}] True: {trueLabel} | Pred: {predLabel} -> "
                    f"Non-Vulnerable: {nonVul:.2f}% | Vulnerable: {vul:.2f}%"
                )
            
            sample_offset += len(inputs)
            
            cm = confusion_matrix(labels.cpu(), predicted.cpu(), labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            accuracy = accuracy_score(labels.cpu(), predicted.cpu())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            total_accuracy += accuracy * len(inputs)
            total_tpr += tpr * len(inputs)
            total_tnr += tnr * len(inputs)
            total_precision += precision * len(inputs)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_f1 += f1 * len(inputs)

    test_acc = total_accuracy / len(X_test)
    test_tpr = total_tpr / len(X_test)
    test_tnr = total_tnr / len(X_test)
    test_precision = total_precision / len(X_test)
    test_avg_f1 = total_f1 / len(X_test)

    # Overall (micro / full-set) precision, recall, and F1 from total counts:
    overall_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    overall_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    overall_f1 = (
        (2 * overall_precision * overall_recall)
        / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    average_ACC += test_acc
    average_TPR += test_tpr
    average_TNR += test_tnr
    average_Precision += test_precision

    print(f"Accuracy: {test_acc:.4f}")
    print(f"TPR / Recall (batch-weighted): {test_tpr:.4f}")
    print(f"TNR: {test_tnr:.4f}")
    print(f"Precision (batch-weighted): {test_precision:.4f}")
    print(f"F1 (batch-weighted): {test_avg_f1:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"tp: {total_tp}")
    print(f"tn: {total_tn}")
    print(f"fp: {total_fp}")
    print(f"fn: {total_fn}")

    # X_test, Y_true = X_test.to(device), Y_true.to(device)
    # with torch.no_grad():
    #     X_test = X_test.permute(0, 3, 1, 2)
    #     outputs = model(X_test)
    #
    #     _, predicted = torch.max(outputs, 1)
    #     cm = confusion_matrix(Y_true.cpu(), predicted.cpu())
    #     tn, fp, fn, tp = cm.ravel()
    #     accuracy = accuracy_score(Y_true.cpu(), predicted.cpu())
    #     tpr = tp / (tp + fn)
    #     tnr = tn / (fn + tn)
    #
    #     average_ACC += accuracy
    #     average_TPR += tpr
    #     average_TNR += tnr
    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"TPR: {tpr:.4f}")
    #     print(f"TNR: {tnr:.4f}")
    #
    #     print(f"tp: {tp}")
    #     print(f"tn: {tn}")
    #     print(f"fp: {fp}")
    #     print(f"fn: {fn}")


def LoadFile(FilePath):
    try:
        return np.load(FilePath)
    except OSError:
        print(f"Skipping corrupted image: {FilePath}")
        return None

def GetDirAllFile(FilePath):
    train_x = []
    train_y = []
    file_names = []
    for root, dirs, files in os.walk(FilePath):
        for file in files:
            path = os.path.join(root, file)
            x = LoadFile(path)
            if x is not None:
                if '_0.npy' in path:
                    train_y.append(0)
                elif '_1.npy' in path:
                    train_y.append(1)
                train_x.append(x)
                file_names.append(file)




    #train_x = torch.stack(train_x)
    #train_y = torch.tensor(train_y, dtype=torch.long)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    file_names = np.array(file_names)
    return train_x, train_y, file_names

def main(TrainFilePath):
    all_evaluate_time = 0
    X, Y, names = GetDirAllFile(TrainFilePath)  # x is the image NumPy array; y is the label NumPy array

    print(f"Loaded X: {len(X)}, Y: {len(Y)} | Class 0: {(Y == 0).sum()},"f" Class 1: {(Y == 1).sum()}")
    
    kf = KFold(5, shuffle=True, random_state=1)


    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        names_test = names[test_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

        model = CNNModel()

        train_model(model, X_train_tensor, Y_train_tensor, train_batch_size, num_epochs)

        start_time = time.time()
        evaluate_model(model, X_test_tensor, Y_test_tensor, eval_batch_size, test_names=names_test)
        end_time = time.time()
        all_evaluate_time += end_time - start_time

    return all_evaluate_time

if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_time = main(TrainFilePath)

    print("-----final-----")
    print("average ACC: {:.4f}".format(average_ACC/5))
    print("average TPR: {:.4f}".format(average_TPR/5))
    print("average TNR: {:.4f}".format(average_TNR/5))
    print("average Precision: {:.4f}".format(average_Precision/5))
    print("evaluate cost time: ", cost_time)