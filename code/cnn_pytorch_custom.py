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
            inputs = inputs.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


# Confusion matrix
def evaluate_model(model, X_test, Y_true, eval_batch_size):
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


    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            cm = confusion_matrix(labels.cpu(), predicted.cpu())
            tn, fp, fn, tp = cm.ravel()
            accuracy = accuracy_score(labels.cpu(), predicted.cpu())
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            precision = tp / (tp + fp)

            total_accuracy += accuracy * len(inputs)
            total_tpr += tpr * len(inputs)
            total_tnr += tnr * len(inputs)
            total_precision += precision * len(inputs)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    average_ACC += total_accuracy / len(X_test)
    average_TPR += total_tpr / len(X_test)
    average_TNR += total_tnr / len(X_test)
    average_Precision = total_precision / len(X_test)
    print(f"Accuracy: {total_accuracy / len(X_test):.4f}")
    print(f"TPR: {total_tpr / len(X_test):.4f}")
    print(f"TNR: {total_tnr / len(X_test):.4f}")
    print(f"Precision: {total_precision / len(X_test):.4f}")

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




    #train_x = torch.stack(train_x)
    #train_y = torch.tensor(train_y, dtype=torch.long)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

def main(TrainFilePath):
    all_evaluate_time = 0
    X, Y = GetDirAllFile(TrainFilePath)  # x is the image NumPy array; y is the label NumPy array
    kf = KFold(5, shuffle=True, random_state=1)


    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

        model = CNNModel()

        train_model(model, X_train_tensor, Y_train_tensor, train_batch_size, num_epochs)

        start_time = time.time()
        evaluate_model(model, X_test_tensor, Y_test_tensor, eval_batch_size)
        end_time = time.time()
        all_evaluate_time += end_time - start_time

    return all_evaluate_time

if __name__=='__main__':

    device = torch.device("cuda")
    cost_time = main(TrainFilePath)

    print("-----final-----")
    print("average ACC: {:.4f}".format(average_ACC/5))
    print("average TPR: {:.4f}".format(average_TPR/5))
    print("average TNR: {:.4f}".format(average_TNR/5))
    print("average Precision: {:.4f}".format(average_Precision/5))
    print("evaluate cost time: ", cost_time)