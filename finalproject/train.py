from collections import namedtuple

import torch
from torchvision import datasets, transforms

import train_cmd_parser
import model_definition

Datasets = namedtuple("Datasets", ["train_set", "val_set", "test_set"])
valid_architectures = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'wide_resnet50_2']
valid_devices = ['cpu', 'cuda', 'mps']

def main():
    print("Possible architectures:")
    print(valid_architectures)

    parser = train_cmd_parser.create_train_parser()
    args = parser.parse_args()

    data_directory = args.data_directory
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    print(gpu)

    if arch not in valid_architectures:
        print(f"Error: {arch} is not a valid architecture.")
        print("Please choose one of the following architectures:")
        print(valid_architectures)
        return

    if gpu not in valid_devices:
        print(f"Error: {gpu} is not a valid device.")
        print("Please choose one of the following devices:")
        print(valid_devices)
        return

    # load datasets
    data_sets = load_datasets(data_directory)
    train_data_loader = data_sets.train_set
    vali_data_loader = data_sets.val_set

    if gpu == 'cpu':
        device = torch.device("cpu")
    else:
        device:str = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

    model = create_model(arch, hidden_units, device, train_data_loader.dataset.class_to_idx)

    optimizer, avg_loss, val_acc = train_model(model, arch, train_data_loader, vali_data_loader, epochs, learning_rate, device)

    save_trained_model(arch, model, save_dir, epochs, avg_loss, optimizer, val_acc)

def save_trained_model(arch: str, model: model_definition.FlowerModel, save_dir, epochs, loss, optimizer, val_acc):
    print("Start saving..")
    torch.save({
        'epoch': epochs,
        'arch': arch,
        'hidden_units': model.hidden_units,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'class_to_idx': model.class_to_idx,
        'accuracy': val_acc
    }, f'{save_dir}/checkpoint.pth')
    print("Saved!")


def train_model(model, model_name, train_data, vali_data, epochs, learning_rate, device):
    print(f"Start training with {model_name}, learning_rate: {learning_rate}, device: {device}, epochs: {epochs}")
    criterion = torch.nn.NLLLoss()            # LogSoftmax
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=learning_rate)
    num_epochs = epochs
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)          # forward
            loss = criterion(outputs, labels)
            loss.backward()                  # backward
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_data)

        # Validation
        model.eval()
        test_correct = 0
        val_total   = 0
        with torch.no_grad():
            for images, labels in vali_data:
                images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100.0 * test_correct / val_total
        print(f'Epoch {epoch+1}/{num_epochs} – loss: {avg_loss:.4f} – val acc: {val_acc:.2f}%')
    print("Training finished.")
    return optimizer, avg_loss, val_acc

def create_model(arch: str, hidden_units: int, device: str, class_to_idx) -> torch.nn.Module:
    model = model_definition.FlowerModel(arch=arch, hidden_units=hidden_units, device=device, class_to_idx=class_to_idx)
    return model

def load_datasets(data_dir: str) -> Datasets:
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transform = transforms.Compose([transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    test_and_validation_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=training_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_and_validation_transform)
    vali_data = datasets.ImageFolder(valid_dir, transform=test_and_validation_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    valiloader = torch.utils.data.DataLoader(vali_data, batch_size=32, shuffle=True)

    return Datasets(train_set=trainloader, val_set=testloader, test_set=valiloader)

if __name__ == '__main__':
    main()

## python train.py "./flower_data" --save_dir "." --arch "resnet101" --learning_rate 0.001 --hidden_units 512 --epochs 6 --gpu mps
## python train.py "./flower_data" --save_dir "." --arch "resnext50_32x4d" --learning_rate 0.001 --hidden_units 512 --epochs 6 --gpu mps
## python train.py "./flower_data" --save_dir "." --arch "wide_resnet50_2" --learning_rate 0.001 --hidden_units 512 --epochs 6 --gpu mps