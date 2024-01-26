#%%
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import PIL.Image as Image
import torch.nn as nn
import numpy as np


#load 28*28 images from MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

num_train = len(trainset)
plt.figure(1)
image, label = trainset[50]
plt.imshow(transforms.ToPILImage()(image))
plt.title(f"Label: {label}")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 16 output channels, 5x5 square convolution kernel
        # input = 1*28*28
        # output = 16*14*14
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        # 16 input image channel, 32 output channels, 5x5 square convolution kernel, stride=2
        # input = 16*14*14
        # output = 32*7*7
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        # linear layer
        # input = 32*7*7
        # output = 10
        self.fc = nn.Linear(32*7*7, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


    
# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader  = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
        

def validation(test_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct, total)


# Train the model

num_batches = len(train_loader)
training_loss_v = []
valid_acc_v = []

#print the first train image as a table of numbers
images, labels = next(iter(train_loader))
print(images[0][0])
print(labels[0])


print("Initial training")
for epoch in range(num_epochs):
    loss_tot = 0
    model.train()
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_tot += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_batches}], Loss: {loss.item():.4f}')
    
    (correct, total) = validation(test_loader, model)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {correct/total:.4f}')
    valid_acc_v.append(correct/total)
    training_loss_v.append(loss_tot)
    
torch.save(model.state_dict(), './model.ckpt')

#%%
# Display weights distribution

def display_weights(weights):
    plt.figure(2)
    plt.subplot(3, 2, 1)
    plt.hist(weights["layer1.0.weight"].cpu().numpy().flatten(), bins=100)
    plt.title("layer1.0.weight")
    plt.subplot(3, 2, 2)
    plt.hist(weights["layer1.0.bias"].cpu().numpy().flatten(), bins=100)
    plt.title("layer1.0.bias")
    plt.subplot(3, 2, 3)
    plt.hist(weights["layer2.0.weight"].cpu().numpy().flatten(), bins=100)
    plt.title("layer2.0.weight")
    plt.subplot(3, 2, 4)
    plt.hist(weights["layer2.0.bias"].cpu().numpy().flatten(), bins=100)
    plt.title("layer2.0.bias")
    plt.subplot(3, 2, 5)
    plt.hist(weights["fc.weight"].cpu().numpy().flatten(), bins=100)
    plt.title("fc.weight")
    plt.subplot(3, 2, 6)    
    plt.hist(weights["fc.bias"].cpu().numpy().flatten(), bins=100)
    plt.title("fc.bias")
    plt.tight_layout()
    plt.show()

weights = model.state_dict()
display_weights(weights)

# Simplify the weights
authorized_weights = [- 0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5]

current = weights["layer1.  0.weight"].cpu().numpy()
for i in range(16):
    for j in range(1):
        for k in range(5):
            for l in range(5):
                current[i][j][k][l] = authorized_weights[np.argmin(np.abs(current[i][j][k][l] - authorized_weights))]
weights["layer1.0.weight"] = torch.from_numpy(current)
current = weights["layer1.0.bias"].cpu().numpy()
for i in range(16):
    current[i] = authorized_weights[np.argmin(np.abs(current[i] - authorized_weights))]
weights["layer1.0.bias"] = torch.from_numpy(current)
current = weights["layer2.0.weight"].cpu().numpy()
for i in range(32):
    for j in range(16):
        for k in range(5):
            for l in range(5):
                current[i][j][k][l] = authorized_weights[np.argmin(np.abs(current[i][j][k][l] - authorized_weights))]
weights["layer2.0.weight"] = torch.from_numpy(current)
current = weights["layer2.0.bias"].cpu().numpy()
for i in range(32):
    current[i] = authorized_weights[np.argmin(np.abs(current[i] - authorized_weights))]
weights["layer2.0.bias"] = torch.from_numpy(current)
current = weights["fc.weight"].cpu().numpy()
for i in range(10):
    for j in range(1568):
        current[i][j] = authorized_weights[np.argmin(np.abs(current[i][j] - authorized_weights))]
weights["fc.weight"] = torch.from_numpy(current)
current = weights["fc.bias"].cpu().numpy()
for i in range(10):
    current[i] = authorized_weights[np.argmin(np.abs(current[i] - authorized_weights))]
weights["fc.bias"] = torch.from_numpy(current)

display_weights(weights)

#%%

# Test the model
test_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
model1 = CNN().to(device)
model1.load_state_dict(weights)
(correct, total) = validation(test_loader, model1)
print(f'Test Accuracy: {correct/total:.4f}')
#%%
#test on 10 images and plot the results
test_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)
outputs = model1(images)
_, predicted = torch.max(outputs.data, 1)
for (image, label, pred) in zip(images, labels, predicted):
    plt.figure(3)
    plt.imshow(transforms.ToPILImage()(image))
    plt.title(f"Label: {label}, Predicted: {pred}")
    plt.show()
print(weights)

# %%
#Save weights as a binary file and as double
weights = model1.state_dict()
f = open("weights.bin", "wb")
for weight in weights["layer1.0.weight"].cpu().numpy().flatten():
    f.write(weight)
for weight in weights["layer1.0.bias"].cpu().numpy().flatten():
    f.write(weight)
for weight in weights["layer2.0.weight"].cpu().numpy().flatten():
    f.write(weight)
for weight in weights["layer2.0.bias"].cpu().numpy().flatten():
    f.write(weight)
for weight in weights["fc.weight"].cpu().numpy().flatten():
    f.write(weight)
for weight in weights["fc.bias"].cpu().numpy().flatten():
    f.write(weight)
f.close()

# read and display the first image of MNIST data/t10k-images.idx3-ubyte

f = open("data/t10k-images.idx3-ubyte", "rb")
f.read(4)
f.read(4)
f.read(4)
f.read(4)

image = [[0 for i in range(28)] for j in range(28)]
for i in range(16):
    f.read(1)
for i in range(28):
    for j in range(28):
        image[i][j] = ((ord(f.read(1)) - 128)/128)
        print(image[i][j])
for i in range(28):
    for j in range(14):
        image[i][j], image[i][14 + j] = image[i][14 + j], image[i][j]
image = np.array(image)
plt.figure(4)
plt.imshow(image)


image = torch.from_numpy(image)
image = image.to(device)
image = image.reshape(1, 1, 28, 28)
image = image.float()
outputs = model1(image)
_, predicted = torch.max(outputs.data, 1)
plt.figure(4)
plt.imshow(transforms.ToPILImage()(image[0][0]))
plt.title(f"Predicted: {predicted}")
plt.show()

# %%
# Nombre de paramètres:
# layer1.0.weight: 16*1*5*5 = 400
# layer1.0.bias: 16
# layer2.0.weight: 32*16*5*5 = 12800
# layer2.0.bias: 32
# fc.weight: 10*32*7*7 = 15680
# fc.bias: 10
# Total: 29238
# Il faut 4 bits pour coder les poids et les biais, donc 29238*4 = 116952 bits = 14.6 ko