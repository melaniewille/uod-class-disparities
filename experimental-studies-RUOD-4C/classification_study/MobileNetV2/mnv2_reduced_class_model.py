## MobileNetV2 Classification Model for Reduced DUO Subsets

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import wandb

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Define the run parameters - CHANGE RUN AND SEED HERE IF DESIRED
# In our set-up we used random seed 45 in run "01", 100 in "02" and 550 in "03" for reproducibility
run_number = "01"
num_epochs = 30
batch_size = 32
learning_rate_init = 0.001
seed = 45          
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Define project structure paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_PATH = os.path.join(project_root, "classification_study", "cls_datasets", "one_class_reduced")
CHECKPOINTS_PATH = os.path.join(project_root, "classification_study", "MobileNetV2", "checkpoints", "checkpoints_one_class_reduced")
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
RESULTS_PATH = os.path.join(project_root, "classification_study", "cls_results", "MobileNetV2", "one_class_reduced")


# Create directories if they don't exist
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Define dataset variations
dataset_variations = [
    f"{BASE_PATH}/0.3_holothurian",
    f"{BASE_PATH}/0.3_echinus",
    f"{BASE_PATH}/0.3_scallop",
    f"{BASE_PATH}/0.3_starfish",
]

# Define classes
CLASS_LIST = ["echinus", "holothurian", "scallop", "starfish"]

# Image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


if __name__ == '__main__': 

    for data_dir in dataset_variations:  # Loop over datasets
        dataset_name = os.path.basename(data_dir)  
        print(f"Training on dataset: {dataset_name}")

        run_name = f"{dataset_name}_run_{run_number}"

        # Initialize W&B
        wandb.init(
            project=f"classification_one_class_reduced",                  
            name=f"{run_name}",                         
            config={                                    
                "optimizer": "Adam",
                "architecture": "ResNet18"
            }
        )
        # wandb.init(mode="disabled") # can uncomment this (and comment out the above lines) if you want to run the code without logging to W&B

        # Check if dataset folders exist
        for split in ['train', 'valid', 'test']:
            path = os.path.join(data_dir, split)
            if not os.path.exists(path):
                print(f"Error - Dataset folder not found: {path}")
                exit(1)

        print("Loading datasets")

        # Load dataset
        train_dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms['train'])
        valid_dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/valid", transform=data_transforms['valid'])
        test_dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/test", transform=data_transforms['test'])

        print(f"Train: {len(train_dataset)} images")
        print(f"Valid: {len(valid_dataset)} images")
        print(f"Test: {len(test_dataset)} images")

        # Data loaders
        wandb.config.batch_size = batch_size  
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Load pre-trained MobileNetV2 model
        print("Loading pre-trained MobileNetV2")
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Modify the classifier for 4-class classification
        model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, len(CLASS_LIST))
        )

        # Set device (GPU if available)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = "cpu"
        model = model.to(device)
        print(f"Using device: {device}")

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_init)
        wandb.config.learning_rate_init = learning_rate_init  # in case of changes to the value of learning_rate_init
        
        # Comment out the below if NOT wanting to use a learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr = 0.00003)  

        # Training parameters
        wandb.config.num_epochs = num_epochs 

        print("Starting training...\n")

        model_name = f"model-{run_name}"
        print(f"Model name: {model_name}")
        wandb.config.model_name = model_name

        best_acc = -1.0
        min_loss = float('inf')

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            step = int(epoch)

            print(f"Epoch {epoch+1}/{num_epochs}...")

            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

                # Print progress after every 10 batches
                if batch_idx % 10 == 0 or batch_idx == len(train_dataloader) - 1:
                    print(f"Batch {batch_idx}/{len(train_dataloader)}: Loss = {loss.item():.4f}")
            
                # Log batch loss
                wandb.log({"Batch Loss": loss.item()})

            train_loss /= len(train_dataloader.dataset)
            train_acc = 100.0 * train_correct / train_total

            # Log epoch metrics
            wandb.log({"Epoch": epoch + 1, "Train Loss": train_loss, "Train Accuracy": train_acc}) 

            print(f"Epoch {epoch+1} Complete: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%\n")

            # Validation step
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            print("Evaluating model on validation set...")

            with torch.no_grad():
                for images, labels in valid_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(valid_dataloader.dataset)
            val_acc = 100.0 * val_correct / val_total

            # Log validation metrics
            wandb.log({"Validation Loss": val_loss, "Validation Accuracy": val_acc})

            print(f"Validation Results: Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%\n")

            # Comment out the below if NOT using a learning rate scheduler:
            scheduler.step(val_loss)
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
            wandb.log({"lr": curr_lr}, step=epoch)

            # save checkpoints
            save_path_acc = os.path.join(f"{CHECKPOINTS_PATH}", f"{run_name}_best_acc.pt")
            save_path_loss = os.path.join(f"{CHECKPOINTS_PATH}", f"{run_name}_best_loss.pt")

            # If the validation accuracy goes up, let's save a checkpoint of our model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path_acc)

            # If the validation loss goes down, let's save a checkpoint of our model
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(model.state_dict(), save_path_loss)

            # We save two checkpoints in case our model does overfitting then we capture 
            # the final version of the model weights before the loss starts going up again

        # Save final model
        save_path_final = os.path.join(f"{CHECKPOINTS_PATH}", f"{run_name}_final.pt")
        torch.save(model.state_dict(), save_path_final)


        # Testing all models
        for checkpoint in ["best_acc", "best_loss", "final"]: 
            model_path = os.path.join(f"{CHECKPOINTS_PATH}", f"{run_name}_{checkpoint}.pt")
            model.load_state_dict(torch.load(model_path))  
            model.eval()
            all_preds = []
            all_labels = []
            print("Running on test set...")

            with torch.no_grad():
                for images, labels in test_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Define results directory
            results_dir = os.path.join(f"{RESULTS_PATH}/{run_name}", f"{run_name}_{checkpoint}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Calculate accuracy, precision, recall, F1
            test_acc = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')

            # Log test accuracy
            wandb.log({f"Accuracy ({checkpoint})": test_acc})
            wandb.log({f"Precision ({checkpoint})": precision})
            wandb.log({f"Recall ({checkpoint})": recall})
            wandb.log({f"F1 Score ({checkpoint})": f1})

            report_str = classification_report(all_labels, all_preds, target_names=CLASS_LIST)
            report_dict = classification_report(all_labels, all_preds, target_names=CLASS_LIST, output_dict=True)

            # Save test metrics
            metrics_path = os.path.join(results_dir, f"evaluation_metrics_{run_name}_{checkpoint}.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"Test Accuracy: {test_acc:.4f}%\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")

                f.write("Per-Class Metrics:\n")
                for cls in CLASS_LIST:
                    f.write(
                        f"{cls:12s}  Precision: {report_dict[cls]['precision']:.4f}  "
                        f"Recall: {report_dict[cls]['recall']:.4f}  "
                        f"F1: {report_dict[cls]['f1-score']:.4f}  "
                        f"Support: {report_dict[cls]['support']}\n"
                    )

            # Compute and log Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  # Convert to percentage
            wandb.log({f"Confusion Matrix ({checkpoint})": wandb.Image(plt)})

            # Save Confusion Matrix (Percentages)
            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=CLASS_LIST)
            disp.plot(cmap=plt.cm.Blues, values_format=".2f")
            plt.title(f"Confusion Matrix - {run_name}_{checkpoint}_model")
            plt.tight_layout(pad=2.0)
            plt.savefig(os.path.join(results_dir, f"confusion_matrix_percentage_{run_name}_{checkpoint}.png"))
            plt.close()

        wandb.finish()