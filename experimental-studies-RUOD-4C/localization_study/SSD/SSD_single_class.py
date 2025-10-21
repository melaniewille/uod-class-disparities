import os
import json
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# Define project structure paths and run parameters
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_PATH = os.path.join(project_root, "localization_study", "loc_datasets", "single_class_data")
CHECKPOINT_PATH = os.path.join(project_root, "localization_study", "SSD", "checkpoints", "checkpoints_single")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
RESULTS_PATH = os.path.join(project_root, "localization_study", "loc_results", "SSD", "single_class")
os.makedirs(RESULTS_PATH, exist_ok=True)

CLASS_LIST = ["echinus", "holothurian", "scallop", "starfish"]

BATCH_SIZE = 8
NUM_EPOCHS = 30
LR = 1e-3
NUM_CLASSES = 2  # background + 1 object
CONF_THRESH = 0.25
IOU_THRESH = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Convert datasets from YOLO
class YoloSingleClassDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + ".txt")

        # Load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id, x_c, y_c, bw, bh = map(float, line.strip().split())
                    # YOLO normalized -> absolute VOC
                    x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
                    x_min = x_c - bw / 2
                    y_min = y_c - bh / 2
                    x_max = x_c + bw / 2
                    y_max = y_c + bh / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)  # single class → always 1

        if len(boxes) == 0:
            # Skip images without boxes
            return None

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            img = self.transform(img)

        return img, target, img_file, (w, h)


# Define helper functions
def collate_fn_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))


def convert_to_coco_json(dataset, save_path, class_name="object"):
    images = []
    annotations = []
    ann_id = 1

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue
        _, target, img_file, (w, h) = sample
        images.append({
            "id": idx,
            "width": w,
            "height": h,
            "file_name": img_file
        })
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box.tolist()
            bw = xmax - xmin
            bh = ymax - ymin
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": 1,
                "bbox": [xmin, ymin, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1

    categories = [{"id": 1, "name": class_name}]
    
    # Add minimal required fields
    coco_dict = {
        "info": {},          # required
        "licenses": [],      # required
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(save_path, "w") as f:
        json.dump(coco_dict, f)



def compute_confusion_matrix(gt, pred, iou_thresh=0.5):
    TP, FP, FN = 0, 0, 0
    matched = []

    for g in gt:
        gbox = g["box"]
        found = False
        for i, p in enumerate(pred):
            if i in matched:
                continue
            pbox = p["box"]
            # IoU
            ixmin = max(gbox[0], pbox[0])
            iymin = max(gbox[1], pbox[1])
            ixmax = min(gbox[2], pbox[2])
            iymax = min(gbox[3], pbox[3])
            iw = max(ixmax - ixmin, 0.)
            ih = max(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((gbox[2]-gbox[0])*(gbox[3]-gbox[1]) +
                   (pbox[2]-pbox[0])*(pbox[3]-pbox[1]) - inters)
            iou = inters / uni if uni > 0 else 0
            if iou >= iou_thresh:
                TP += 1
                matched.append(i)
                found = True
                break
        if not found:
            FN += 1

    FP = len(pred) - len(matched)
    TN = 0  # not meaningful for detection
    return TP, FP, FN, TN



# Main loop
transform = T.Compose([T.ToTensor()])

for class_name in CLASS_LIST:
    TRAIN_IMG_DIR = os.path.join(BASE_PATH, class_name, "images/train")
    TRAIN_LABEL_DIR = os.path.join(BASE_PATH, class_name, "labels/train")
    VAL_IMG_DIR = os.path.join(BASE_PATH, class_name, "images/valid")
    VAL_LABEL_DIR = os.path.join(BASE_PATH, class_name, "labels/valid")
    TEST_IMG_DIR = os.path.join(BASE_PATH, class_name, "images/test")
    TEST_LABEL_DIR = os.path.join(BASE_PATH, class_name, "labels/test")

    # Datasets
    train_dataset = YoloSingleClassDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform)
    val_dataset   = YoloSingleClassDataset(VAL_IMG_DIR, VAL_LABEL_DIR, transform)
    test_dataset  = YoloSingleClassDataset(TEST_IMG_DIR, TEST_LABEL_DIR, transform)

    # COCO JSONs for evaluation
    val_ann_file = os.path.join(CHECKPOINT_PATH, f"{class_name}_val.json")
    test_ann_file = os.path.join(CHECKPOINT_PATH, f"{class_name}_test.json")
    convert_to_coco_json(val_dataset, val_ann_file, class_name)
    convert_to_coco_json(test_dataset, test_ann_file, class_name)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_skip_none)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_skip_none)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_skip_none)

    # Load model
    print(f"Setting up model for {class_name}")
    model = ssd300_vgg16(pretrained=True)
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head = SSDHead(in_channels, num_anchors, NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    best_loss = float("inf")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n========== Training {class_name} ==========\n")
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{class_name} Epoch {epoch+1}/{NUM_EPOCHS}"):
            # Skip empty batch
            if len(batch) == 0:
                continue

            # batch is a tuple of lists: (images, targets, img_files, whs)
            if len(batch) != 4:
                continue

            images_list, targets_list, _, _ = batch

            # Filter None samples
            filtered = [(img, tgt) for img, tgt in zip(images_list, targets_list) if img is not None and tgt is not None]
            if len(filtered) == 0:
                continue

            images, targets = zip(*filtered)
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Forward + loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f"ssd_best_{class_name}.pth"))
            print(f"Saved new best model for {class_name}")

    # Evaluation loop
    print(f"Loading best model for testing {class_name}")

    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, f"ssd_best_{class_name}.pth")))
    model.eval()

    coco_gt = COCO(test_ann_file)
    predictions = []
    all_gt = []
    all_pred = []

    print(f"\n========== Testing {class_name} ==========\n")
    
    class_results_path = os.path.join(RESULTS_PATH, class_name)
    os.makedirs(class_results_path, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {class_name}"):
            # batch is a tuple of lists (images, targets, img_files, whs)
            if len(batch) != 4:
                continue  # skip invalid batch
            images_list, targets_list, img_files_list, whs_list = batch

            # Remove None entries
            filtered = [(img, tgt, fname, wh) for img, tgt, fname, wh in zip(images_list, targets_list, img_files_list, whs_list) if img is not None and tgt is not None]
            if len(filtered) == 0:
                continue

            images, targets, img_files, whs = zip(*filtered)
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = targets[i]["image_id"].item()

                # Ground truth boxes
                for b in targets[i]["boxes"].cpu().numpy():
                    all_gt.append({"box": b})

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                # Filter by confidence threshold
                keep = scores >= CONF_THRESH
                boxes, scores = boxes[keep], scores[keep]

                for box, score in zip(boxes, scores):
                    xmin, ymin, xmax, ymax = box
                    w, h = xmax - xmin, ymax - ymin
                    predictions.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [float(xmin), float(ymin), float(w), float(h)],
                        "score": float(score)
                    })
                    all_pred.append({"box": [xmin, ymin, xmax, ymax], "score": float(score)})

    # Save predictions JSON
    pred_file = os.path.join(class_results_path, f"predictions.json")
    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    # COCO mAP evaluation
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP_50_95 = coco_eval.stats[0]
    mAP_50 = coco_eval.stats[1]

    # Confusion matrix calculation
    TP, FP, FN, TN = compute_confusion_matrix(all_gt, all_pred, iou_thresh=IOU_THRESH)
    confusion_stats = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

    metrics = {
        "mAP@0.5": mAP_50,
        "mAP@0.5:0.95": mAP_50_95,
        "confusion_matrix": confusion_stats
    }

    # Save metrics JSON
    metrics_file = os.path.join(class_results_path, f"evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Finished evaluation for {class_name}")
    print(f"Saved predictions → {pred_file}")
    print(f"Saved metrics → {metrics_file}")
