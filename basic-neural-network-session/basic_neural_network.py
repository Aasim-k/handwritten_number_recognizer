import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F


train_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])


train_data = datasets.MNIST('../data', train=True, download=False, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=False, transform=test_transforms)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Block 1: Conv + BN + ReLU
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),    # [28→26]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Block 2: Conv + BN + ReLU + MaxPool + Dropout
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1),   # [26→24]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),           # [32,12,12]
            nn.Dropout(0.1)
        )

        # Block 3: Conv + BN + ReLU
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),   # [12→10]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Dropout before FC
        self.dropout_fc = nn.Dropout(0.05)

        # Fully connected
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.block1(x)   # -> [16,26,26]
        x = self.block2(x)   # -> [32,12,12]
        x = self.block3(x)   # -> [64,10,10]

        # GAP
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # -> [64]

        # Dropout + FC
        x = self.dropout_fc(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data, target
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data, target

            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)

            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print(f"Test set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({100. * correct / len(test_loader.dataset):.2f}%)\n")


model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
criterion = nn.CrossEntropyLoss()

summary(model, input_size=(1, 28, 28))

num_epochs = 1
for epoch in range(1, num_epochs+1):
    print(f'Epoch {epoch}')
    train(model, None, train_loader, optimizer, criterion)
    test(model, None, test_loader, criterion)   # <-- use test_loader
    scheduler.step()

