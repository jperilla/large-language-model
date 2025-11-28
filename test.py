import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())


def to_onehot(y, num_classes):
    """
    Function for one-hot encoding.
    """
    y_onehot = torch.zeros(y.size(0), num_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot


calced_sha256, calced_md5 = "", ""

y = torch.tensor([0, 1, 2, 2])
y_enc = to_onehot(y, 3)

print('one-hot encoding:\n', y_enc) # one-hot encoding

# Types
print(y.dtype)
x = y.to(torch.float32) # Cast to 32-bit, better for GPU computation
print(x.dtype)

# Size and shape
print(torch.Size(y)) # Check the size
print(y.reshape([4, 1])) # Reshape into new size
print(y.view([1,4]))
print(y.view([4,1])) # Same as reshape, but requires data to be contiguous

# Multiply - must be a square dimension
tensor2d = torch.tensor([[1,2], [3,4]])
z = tensor2d.matmul(tensor2d)
print(f'z = {z}')
print(tensor2d @ tensor2d) # Alternate


# differentiation engine - autograd - holds a directed graph for computational execution
# logistic regression to calculate loss
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, create_graph=True)
grad_L_b = grad(loss, b, create_graph=True)

print(grad_L_w1)
print(grad_L_b)

# Loading data into datasets
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5],
])
y_train = torch.tensor([0,0,0,1,1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0,1])

# Custom dataset class
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))

# Sample with dataloader
from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

# Example training of toy dataset
import torch.nn.functional as F

torch.manual_seed(123)
model = nn.NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()

for batch_idx, (features, labels) in enumerate(train_loader):
    logits = model(features)

    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Logging
    print(f"Epoch: {epoch + 1: 03d}/{num_epochs: 03d}"
          f"Batch {batch_idx:03d}/{len(train_loader):03d}"
          f"Train Loss: {loss:2f}")