# Step 6: Load the Dataset and Define Data Loaders

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    batch_size: int,
    ):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.Resize((256, 256)),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225]
                                                            )])

    test_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_datasets, batch_size=batch_size)
    validloader = DataLoader(valid_datasets, batch_size=batch_size)
    
    return trainloader, testloader, validloader
