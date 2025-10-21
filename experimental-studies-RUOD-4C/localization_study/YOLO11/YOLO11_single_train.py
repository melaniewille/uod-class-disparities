from ultralytics import YOLO
import torch
import wandb
import os

if __name__ == "__main__":

    # Define project structure paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    BASE_PATH = os.path.join(project_root, "localization_study", "loc_datasets", "single_class_data")
    CHECKPOINT_PATH = os.path.join(project_root, "localization_study", "YOLO11", "checkpoints")
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    print(f"Base path: {BASE_PATH}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")

    # Classes for single_class training
    CLASS_LIST = ["echinus", "holothurian", "scallop", "starfish"]

    # Loop over datasets
    for class_name in CLASS_LIST:
        # Define the run parameters
        epoch_number = 30
        run_name = f"{class_name}"   
        project_name = "single_class"

        images_dir = os.path.join(BASE_PATH, class_name, "images")
        labels_dir = os.path.join(BASE_PATH, class_name, "labels")
        yaml_path = os.path.join(BASE_PATH, class_name, f"temp_data_{class_name}.yaml")

        with open(yaml_path, "w") as f:
            f.write(f"""
        train: {os.path.join(images_dir, "train")}
        val: {os.path.join(images_dir, "valid")}
        nc: 1
        names: ["{class_name}"]
        """)

        # Set device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using mps as the default device.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using cuda as the default device.")
        else:
            device = torch.device("cpu")
            print("Using cpu as the default device.")

        # Load YOLOv11 model and move it to the chosen device
        model = YOLO("yolo11n.pt")  # Use pretrained model
        model = model.to(device)

        # Before model training: best model is saved per default in "current_directory/project_name/run_name/weights/best.pt"
        # Set current working directory to the checkpoint path
        current_directory = os.chdir(CHECKPOINT_PATH)
        
        # Train the model
        results_train = model.train(data=yaml_path,
                                    epochs=epoch_number, 
                                    batch=8,
                                    device=device,
                                    project=project_name,                         
                                    name=f"{run_name}")

        # Finish W&B logging
        wandb.finish()