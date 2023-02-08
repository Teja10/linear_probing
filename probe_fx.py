import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('resnet_final.pt').to(device)

train_nodes, eval_nodes = get_graph_node_names(model)

# filter all conv and fc layers in eval_nodes
filtered = filter(lambda x: 'conv' in x or 'fc' in x, eval_nodes)
nodes = list(filtered)

return_nodes = {v: f'layer{k}' for k, v in enumerate(nodes)}
fx = create_feature_extractor(model, return_nodes)
with torch.no_grad():
    out = fx(torch.randn(1, 3, 32, 32).to(device))

features = [t.view(1,-1) for t in out.values()]
probes = [nn.Linear(f.shape[1], 10).to(device) for f in features]
optims = [torch.optim.Adam(p.parameters(), lr=0.001) for p in probes]

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)
num_epochs = 40
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = fx(images)
        features = [t.view(t.shape[0], -1) for t in out.values()]
        for j, (f, p, o) in enumerate(zip(features, probes, optims)):
            out = p(f)
            loss = nn.CrossEntropyLoss()(out, labels)
            o.zero_grad()
            loss.backward()
            o.step()
            if (i+1) % 100 == 0:
                print(f'Probe [{j+1}/{len(probes)}], Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)