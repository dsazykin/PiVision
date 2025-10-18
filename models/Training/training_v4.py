# ====== train_v3_ddp.py ======

import os, random, sys, time
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import mediapipe as mp
import cv2
from PIL import Image

# ---------- Settings ----------
base_dir   = "/home/jovyan/hagrid_30k_hands"
original_dir = "/home/jovyan/.cache/kagglehub/datasets/innominate817/hagrid-sample-30k-384p/versions/5/hagrid-sample-30k-384p/hagrid_30k"
epochs     = 30
batch_size = 32
lr         = 1e-4

# ---------- Utility ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

def log0(msg, *, flush=True):
    """Print only from rank 0"""
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=flush)

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.GaussianBlur(3, (0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- Image Processing ----------
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def crop_hand(image_path, save_path=None, output_size=224):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(img_rgb)

    h, w, _ = img.shape
    if not results.multi_hand_landmarks:
        # fallback to original full image
        resized = cv2.resize(img, (output_size, output_size))
    else:
        hand_landmarks = results.multi_hand_landmarks[0]
        xs = [int(p.x * w) for p in hand_landmarks.landmark]
        ys = [int(p.y * h) for p in hand_landmarks.landmark]
        x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
        y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)
        cropped = img[y_min:y_max, x_min:x_max]
        resized = cv2.resize(cropped, (output_size, output_size))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, resized)

    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

def process_images():
    print("⏳ Preprocessing images and cropping hands (with fallback)...")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    total = 0
    processed = 0

    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        out_dir = os.path.join(base_dir, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            total += 1
            img_path = os.path.join(class_dir, img_name)
            save_path = os.path.join(out_dir, img_name)
            if os.path.exists(save_path):
                continue  # skip already processed
            cropped = base_dir(img_path, save_path)
            if cropped:
                processed += 1

    print(f"✅ Cropping complete: {processed}/{total} images processed.")

# ---------- DDP Initialization ----------
def init_ddp():
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank        = int(os.environ["RANK"])
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

# ---------- Class Weights ----------
def build_class_weights(train_subset, num_classes):
    labels = [train_subset.dataset.samples[i][1] for i in train_subset.indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    w = inv / inv.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)

# ---------- Training ----------
def train():
    set_seed(42)
    rank, local_rank, world = init_ddp()
    device = torch.device(f"cuda:{local_rank}")

    log0(f"[DDP] world_size={world}, rank={rank}, local_rank={local_rank}")
    torch.cuda.synchronize()

    # Dataset + split
    full = datasets.ImageFolder(base_dir, transform=transform)
    num_classes = len(full.classes)
    n = len(full)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(123))

    # Distributed samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    log0(f"[Data] train_len={len(train_ds)} val_len={len(val_ds)} "
         f"train_steps={len(train_dl)} val_steps={len(val_dl)} batch={batch_size}")

    # ---------- Model ----------
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ---------- Loss + Optimizer + Scheduler ----------
    cls_w = build_class_weights(train_ds, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_val = 0.0
    log0("[Train] Starting training...")

    # ---------- Epoch Loop ----------
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        start_epoch = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, (imgs, labels) in enumerate(train_dl, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            # progress every 100 batches
            if step % 100 == 0 and rank == 0:
                pct = 100 * step / len(train_dl)
                print(f"[Epoch {epoch+1}] Step {step}/{len(train_dl)} ({pct:.1f}%) "
                      f"loss={loss.item():.4f}", flush=True)

        # ---------- Validation ----------
        torch.distributed.barrier()
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # ---------- Reduce metrics ----------
        tensors = {
            "train_loss": torch.tensor([total_loss], device=device),
            "train_batches": torch.tensor([len(train_dl)], device=device),
            "train_acc": torch.tensor([correct, total], dtype=torch.long, device=device),
            "val_acc": torch.tensor([val_correct, val_total], dtype=torch.long, device=device),
        }
        for t in tensors.values():
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

        avg_loss = (tensors["train_loss"] / tensors["train_batches"]).item()
        train_acc = 100.0 * tensors["train_acc"][0].item() / max(tensors["train_acc"][1].item(), 1)
        val_acc   = 100.0 * tensors["val_acc"][0].item() / max(tensors["val_acc"][1].item(), 1)

        torch.distributed.barrier()
        if rank == 0:
            epoch_sec = time.time() - start_epoch
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss {avg_loss:.4f} | Train {train_acc:.2f}% | "
                  f"Val {val_acc:.2f}% | {epoch_sec/60:.1f} min", flush=True)

            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.module.state_dict(), "gesture_model_v4_handcrop.pth")

        scheduler.step()

    log0("[Train] Finished training.")

    # Export to ONNX only from rank 0
    if rank == 0:
        log0("[Export] Saving gesture_model_v4_handcrop.pth and gesture_model_v4_handcrop.onnx ...")

        # Rebuild model on CPU
        export_model = models.efficientnet_b0(weights=None)
        in_f = export_model.classifier[1].in_features
        export_model.classifier[1] = nn.Linear(in_f, num_classes)
        export_model.load_state_dict(torch.load("gesture_model_v4_handcrop.pth", map_location="cpu"))
        export_model.eval()

        # Dummy input for ONNX tracing (batch size 1, 3x224x224 image)
        dummy = torch.randn(1, 3, 224, 224, device="cpu")

        torch.onnx.export(
            export_model, dummy, "gesture_model_v4_handcrop.onnx",
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17
        )

        log0("[Export] ONNX model saved successfully: gesture_model_v4_handcrop.onnx")

    torch.distributed.destroy_process_group()

# ---------- Entry ----------
if __name__ == "__main__":
    train()