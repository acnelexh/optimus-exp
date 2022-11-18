import torch
import torchvision
import pandas as pd
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from pytorch_resnet_cifar10.resnet import resnet20, resnet32, resnet44, resnet56

model_fns = [resnet18, resnet20, resnet32, resnet44, resnet56]

def _train_model_step(model, data, label, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, label)
    loss.backward()
    optimizer.step()
    return loss

def _train_model(
    model,
    train_loader,
    optimizer,
    epochs=350,
    device='cuda'):
    step_count = 0
    step_loss = []
    for e in range(epochs):
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            loss = _train_model_step(model, data, label, optimizer)
            step_loss.append(loss)
            step_count += 1
            if step_count % 100 == 0:
                print(f'Epoch: {e}, Step: {step_count}, Loss: {loss}')
    return step_loss
    
def train_model(epochs=350, device='cuda'):
    batch_size = 128
    with open('results.csv', 'w') as f:
        df = pd.DataFrame()
    for idx, model_fn in enumerate(model_fns):
        model = model_fn()
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        cifar_loader = get_cifar_loader(batch_size)
        step_loss = _train_model(model, cifar_loader, optimizer, epochs)
        if idx == 0:
            df.insert(0, 'step', range(len(step_loss)))
        step_loss_detach = [x.cpu().detach().numpy() for x in step_loss]
        df.insert(idx+1, model_fn.__name__, step_loss_detach)
    df.to_csv('results.csv')
    return
        

def get_cifar_loader(batch_size=128):
    cifar_dataset = CIFAR10(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor())
    cifar_loader = torch.utils.data.DataLoader(
        cifar_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    return cifar_loader

def main():
    train_model(1)

if __name__ == "__main__":
    main()