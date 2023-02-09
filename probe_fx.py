import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import wandb

wandb.init(
    project='pytorch-probes',

    config={
        'epochs': 40,
        'batch_size': 100,
        'learning_rate': 0.001,
        'architecture': 'resnet18',
        'dataset': 'cifar10'
    }
)

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
# step = 0
# steps = []
# probe_losses = [[] for _ in range(len(probes))]
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = fx(images)
        features = [t.view(t.shape[0], -1) for t in out.values()]
        probe_vals = {}
        for j, (f, p, o) in enumerate(zip(features, probes, optims)):
            out = p(f)
            loss = nn.CrossEntropyLoss()(out, labels)
            o.zero_grad()
            loss.backward()
            o.step()
            if (i+1) % 100 == 0:
                print(f'Probe [{j+1}/{len(probes)}], Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            probe_vals[f'Probe {j+1} loss'] = loss.item()
            # probe_losses[j].append(loss.item())
        wandb.log(probe_vals)
        # steps.append(step)
        # step += 1

        # wandb.log({"Probe Losses" : wandb.plot.line_series(
        #                xs=steps, 
        #                ys=probe_losses,
        #                keys=nodes,
        #                title="Probe Losses",
        #                xname="Step")})


test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = [0 for _ in range(len(probes))]
    total = [0 for _ in range(len(probes))]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = fx(images)
        features = [t.view(t.shape[0], -1) for t in out.values()]
        probe_vals = {}
        for j, (f, p, o) in enumerate(zip(features, probes, optims)):
            out = p(f)
            _, predicted = torch.max(out.data, 1)
            total[j] += labels.size(0)
            correct[j] += (predicted == labels).sum().item()
            wandb.run.summary[f"Probe {j+1} accuracy"] = 100 * correct[j] / total[j]
        # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
