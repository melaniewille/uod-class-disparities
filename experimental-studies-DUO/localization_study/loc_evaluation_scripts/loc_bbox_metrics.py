import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import euclidean


# Define project structure paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = project_root
RESULTS_PATH = os.path.join(project_root, "loc_results")


# Parameters
MODELS = ["YOLO11", "SSD"]
CLASS_LIST = ["scallop", "echinus", "holothurian", "starfish"]
EXPERIMENTS = ["single_class", "balanced", "0.75_reduced", "0.5_reduced", "0.25_reduced"]
results = []

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area != 0 else 0

def compute_center(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)

def evaluate(gt_boxes_by_img, predictions):
    ious, center_errors, width_errors, height_errors, area_errors = [], [], [], [], []

    for pred in predictions:
        pred_box = pred['bbox']
        gt_boxes = gt_boxes_by_img.get(pred['image_id'], [])
        if not gt_boxes:
            continue

        best_iou = 0
        best_gt = None
        for gt_box in gt_boxes:
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_box

        if best_gt:
            ious.append(best_iou)
            center_errors.append(euclidean(compute_center(pred_box), compute_center(best_gt)))
            pw, ph = pred_box[2], pred_box[3]
            gw, gh = best_gt[2], best_gt[3]
            width_errors.append(abs(pw - gw) / gw * 100 if gw != 0 else 0)
            height_errors.append(abs(ph - gh) / gh * 100 if gh != 0 else 0)
            area_errors.append(abs(pw * ph - gw * gh) / (gw * gh) * 100 if gw * gh != 0 else 0)


    return {
        "average_iou": np.mean(ious) if ious else 0,
        "average_center_error": np.mean(center_errors) if center_errors else 0,
        "average_width_error": np.mean(width_errors) if width_errors else 0,
        "average_height_error": np.mean(height_errors) if height_errors else 0,
        "average_area_error": np.mean(area_errors) if area_errors else 0
    }

# Main loop
for class_name in CLASS_LIST:
    gt_file = f"{BASE_PATH}/loc_evaluation_scripts/ground_truths/ground_truth_test_{class_name}.json"
    if not os.path.exists(gt_file):
        print(f"Missing GT file for {class_name}")
        continue
    with open(gt_file) as f:
        gt_data = json.load(f)

    gt_boxes_by_img = defaultdict(list)
    for ann in gt_data["annotations"]:
        gt_boxes_by_img[ann["image_id"]].append(ann["bbox"])

    for exp in EXPERIMENTS:
        for model in MODELS:
            pred_file = f"{RESULTS_PATH}/{model}/{exp}/{class_name}/predictions.json"
            if not os.path.exists(pred_file):
                print(f"Missing prediction file: {pred_file}")
                continue
            with open(pred_file) as f:
                predictions = json.load(f)

            metrics = evaluate(gt_boxes_by_img, predictions)
            results.append({
                "class": class_name,
                "experiment": exp,
                "model": model,
                **metrics
                })

# Save results
df = pd.DataFrame(results)
output_path = f"{RESULTS_PATH}/bbox_metrics_report.csv"
df.to_csv(output_path, index=False)
print("Evaluation complete.")
