import torch
import wandb as wb
import os
import json 
import pandas as pd

from ultralytics import YOLO

from yolo11_object_detection.logging_helpers import on_val_start, on_val_end

if __name__ == "__main__":
   
    # Define project structure paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    BASE_PATH = os.path.join(project_root, "localization_study", "loc_datasets", "reduced_data")
    CHECKPOINT_PATH = os.path.join(project_root, "localization_study", "YOLO11", "checkpoints")
    RESULTS_PATH = os.path.join(project_root, "localization_study", "loc_results", "YOLO11")
    os.makedirs(RESULTS_PATH, exist_ok=True)
    print(f"Results path: {RESULTS_PATH}")

    # Classes for reduced training
    CLASS_LIST = ["echinus", "holothurian", "scallop", "starfish"]

    # Reduction factors for reduced training
    REDUCTION_LEVEL = ["0.75", "0.25", "0.5"]

    for class_name in CLASS_LIST:
        for reduction_factor in REDUCTION_LEVEL:
            print(f"Testing on: {class_name} for {reduction_factor} data")
            # Define the run parameters (make sure identical to train run)
            run_name = f"{class_name}"   
            project_name = f"{reduction_factor}_reduced"

            images_dir = os.path.join(BASE_PATH, f"{reduction_factor}_labels", class_name, "images")
            labels_dir = os.path.join(BASE_PATH, f"{reduction_factor}_labels", class_name, "labels")
            yaml_path = os.path.join(BASE_PATH, f"{reduction_factor}_labels", class_name, f"temp_data_{class_name}.yaml")

            with open(yaml_path, "w") as f:
                f.write(f"""
            train: {os.path.join(images_dir, "train")}
            val: {os.path.join(images_dir, "valid")}
            test: {os.path.join(images_dir, "test")} 
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

            # Test saved model
            model_path = os.path.join(f"{CHECKPOINT_PATH}/{project_name}/{run_name}/weights/best.pt")
            model = YOLO(model_path)
            model.add_callback("on_val_start", on_val_start)
            model.add_callback("on_val_end", on_val_end)
            model = model.to(device)

            # Before model training: results are saved per default in "current_directory/project_name/run_name"
            # Set current working directory to the results path
            current_directory = os.chdir(RESULTS_PATH)

            # Test the model
            results_test = model.val(data=yaml_path, 
                                    save_json=True,
                                    conf=0.25,
                                    iou=0.7,
                                    device=device,
                                    split="test",
                                    project=project_name,
                                    name=f"{run_name}",
                                    save_txt=True,
                                    save_conf=True)

        # Finish W&B logging
        wb.finish()