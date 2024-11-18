# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="aml_lab1_1",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "CNN",
    "dataset": "FashionMNIST",
    "epochs": 50,
    }
)

# %%
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x.view(-1, 28 * 28))
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# %%
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# %%
def GetModelAndResults(train_dataset, test_dataset, type="linear", epochs=50):
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
  if type == "linear":
    model = SimpleNet()
  elif type == "cnn":
    model = CnnNet()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  for epoch in range(epochs):
    # Training phase
    model.train()
    training_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        # Compute accuracy (optional)
        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    training_loss /= len(train_loader)  # Average training loss
    training_accuracy = total_correct / total_samples * 100

    # Validation phase
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

    validation_loss /= len(test_loader)  # Average validation loss

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "training_accuracy": training_accuracy
    })

    print(f"Epoch {epoch+1}: Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%")

  # Evaluation
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for images, labels in test_loader:
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f'Accuracy: {accuracy}%')
  # Save your script file as an artifact
  artifact = wandb.Artifact("training_script", type="code")
  artifact.add_file("AML_Lab1.ipynb")  # Replace with the name of your script
  wandb.log_artifact(artifact)
  wandb.finish()
  return model, accuracy

# %%
# wandb.init(
#     project="aml_lab1_1",
#     config={
#     "learning_rate": 0.01,
#     "architecture": "linear",
#     "dataset": "FashionMNIST",
#     "epochs": 50,
#     }
# )
# linearModel, linearAccuracy = GetModelAndResults(train_dataset, test_dataset, type="linear", epochs=50)

# wandb.init(
#     project="aml_lab1_1",
#     config={
#     "learning_rate": 0.01,
#     "architecture": "CNN",
#     "dataset": "FashionMNIST",
#     "epochs": 50,
#     }
# )
cnnModel, cnnAccuracy = GetModelAndResults(train_dataset, test_dataset, type="cnn", epochs=50)

# %%
