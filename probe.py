import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchviz import make_dot

#load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('resnet_final.pt').to(device)
# Exclude subgraphs for feature extraction
for param in model.parameters():
    param.requires_grad = False

def extract_sizes(x, model):
    input_features = []
    output_features = []

    def hook(module, input, output):
        input_features.append(input[0].view(input[0].shape[0], -1))
        output_features.append(output)

    # Set the model to evaluation mode
    model.eval()

    # Register a forward hook on each module
    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.register_forward_hook(hook)

    # Use the model on a single input tensor
    model(x)

    return input_features, output_features

# Extract features from an input tensor
x = torch.randn(1, 3, 32, 32).to(device)
input_features, output_features = extract_sizes(x, model)

# Print the shape of the features at each layer
for i, f in enumerate(input_features):
    print(f"Input: {f.shape} | Output: {output_features[i].shape}")

def extract_features(input_batch, model):
    with torch.no_grad():
        input_features = []
        
        def hook(module, input, output):
            input_features.append(input[0].detach())
        
        model.eval()
        hooks = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook))
        with torch.no_grad():
            model(input_batch)
        # remove all hooks
        for hook in hooks:
            hook.remove()
        return input_features

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

# Create list of probes and optimizers
probes = []
for i in range(len(input_features)):
    probes.append(nn.Linear(input_features[i].shape[1], 10).to(device))

probe_optimizers = []
for probe in probes:
    probe_optimizers.append(torch.optim.Adam(probe.parameters(), lr=0.001))

loss_fns = [nn.CrossEntropyLoss() for i in range(len(input_features))]

def train_probes(input_features, probes, probe_optimizers, loss_fn, labels):
    for i, input in enumerate(input_features):
        out = probes[i](input.view(input.shape[0], -1))
        loss = loss_fn[i](out, labels)
        probe_optimizers[i].zero_grad()
        loss.backward()
        probe_optimizers[i].step()
            
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            input_features = extract_features(images, model)
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
        train_probes(input_features, probes, probe_optimizers, loss_fns, labels)
        print(torch.cuda.memory_summary(device=device, abbreviated=True))
    print('Epoch {i} Done')