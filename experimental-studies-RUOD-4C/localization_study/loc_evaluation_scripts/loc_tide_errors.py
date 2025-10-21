import json
import os
import io
import sys
import tempfile
from collections import Counter
from tidecv import TIDE
import tidecv.datasets as datasets
from tidecv import (
    ClassError,
    BoxError,
    DuplicateError,
    BackgroundError,
    OtherError,
    MissedError,
)


def make_box_poly_from_bbox(bbox):
    x, y, w, h = bbox
    return [[
        x,     y,
        x + w, y,
        x + w, y + h,
        x,     y + h
    ]]

def patch_coco_json(orig_json_path):
    data = json.load(open(orig_json_path))
    for ann in data.get("annotations", []):
        seg = ann.get("segmentation")
        if (
            (not isinstance(seg, list))
            or (not seg)
            or isinstance(seg[0], (int, float))
        ):
            ann["segmentation"] = make_box_poly_from_bbox(ann["bbox"])
    fd, fixed_path = tempfile.mkstemp(
        suffix=".json", prefix="patched_", dir=os.path.dirname(orig_json_path)
    )
    os.close(fd)
    with open(fixed_path, "w") as f:
        json.dump(data, f)
    print(f"Patched JSON written to: {fixed_path}")
    return fixed_path

def run_tide_evaluation(orig_ann, resFile, tag, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Load Data
    fixed_ann = patch_coco_json(orig_ann)
    gt = datasets.COCO(fixed_ann)
    pred = datasets.COCOResult(resFile)

    tide = TIDE()
    res = tide.evaluate(gt, pred, mode=TIDE.BOX, pos_threshold=0.25)

    # Save TIDE Summary (dAP)
    summary = os.path.join(out_dir, f"{tag}_tide_summary.txt")
    buf = io.StringIO()
    sys.stdout = buf
    tide.summarize()
    sys.stdout = sys.__stdout__

    with open(summary, "w") as f:
        f.write(buf.getvalue())
    print(f"Summary saved to: {summary}")

    # Save Plots
    tide.plot(out_dir)
    print(f"Plots saved to: {out_dir}")

    # Count errors (absolute)
    err_counter = Counter(type(err).__name__ for err in res.errors)

    # Print absolute counts
    for err_name, count in err_counter.items():
        print(f"{err_name}: {count}")

    # Save per-error-type counts
    counts = os.path.join(out_dir, f"{tag}_error_counts.txt")
    with open(counts, "w") as f:
        f.write(f"Error Type Counts for {tag}\n\n")
        for err_name, count in err_counter.items():
            line = f"{err_name}: {count}\n"
            f.write(line)
    print(f"Error counts saved to: {counts}")


if __name__ == "__main__":

    # Define project structure paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    BASE_PATH = project_root
    RESULTS_PATH = os.path.join(project_root, "loc_results")

    MODELS = ["YOLO11", "SSD"]
    CLASS_LIST = ["scallop", "echinus", "holothurian", "starfish"]
    EXPERIMENTS = ["single_class", "balanced", "0.75_reduced", "0.5_reduced", "0.25_reduced"]

    for class_name in CLASS_LIST:
        gt_file = f"{BASE_PATH}/loc_evaluation_scripts/ground_truths/ground_truth_test_{class_name}.json"
        if not os.path.exists(gt_file):
            print(f"Missing GT file: {gt_file}")
            continue
        for exp in EXPERIMENTS:
            for model in MODELS:
                pred_file = f"{RESULTS_PATH}/{model}/{exp}/{class_name}/predictions.json"
                if not os.path.exists(pred_file):
                    print(f"Missing prediction file: {pred_file}")
                    continue
                out_dir = f"{RESULTS_PATH}/{model}/{exp}/{class_name}/TIDE"
                tag = f"{model}_{exp}_{class_name}"
                print(f"Running: {tag}")
                run_tide_evaluation(gt_file, pred_file, tag, out_dir)
    print("All TIDE evaluations completed.")
