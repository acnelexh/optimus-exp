import torch
import torchvision
import pandas as pd
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from pytorch_resnet_cifar10.resnet import resnet20, resnet32, resnet44, resnet56
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip, Normalize


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
        scheuler,
        epochs=350,
        device='cuda'):
    step_count = 0
    step_loss = []
    for e in range(epochs):
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            loss = _train_model_step(model, data, label, optimizer)
            scheuler.step()
            step_loss.append(loss)
            step_count += 1
            if step_count % 100 == 0:
                print(f'Epoch: {e}, Step: {step_count}, Loss: {loss}')
    return step_loss
    
def train_model(
        model,
        train_loader,
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        epochs=350,
        T_max=200,
        device='cuda',
        output_file='output.csv'):
    # main training function
    batch_size = 128
    df = pd.DataFrame()
    model_fn = model_fns[1]
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheuler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    step_loss = _train_model(model, train_loader, optimizer, scheuler, epochs=epochs)
    df.insert(0, 'step', range(len(step_loss)))
    step_loss_detach = [x.cpu().detach().numpy() for x in step_loss]
    df.insert(1, model_fn.__name__, step_loss_detach)
    df.to_csv(output_file, index=False)
    return
        

def get_cifar_loader(batch_size=128):
    transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    cifar_dataset = CIFAR10(
        './data',
        train=True,
        download=True,
        transform=transform)
    cifar_loader = torch.utils.data.DataLoader(
        cifar_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    return cifar_loader

def main():
    model = model_fns[1]()
    trainLoader = get_cifar_loader(128)
    train_model(
        model,
        trainLoader,
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        epochs=350,
        T_max=200)

if __name__ == "__main__":
    main()