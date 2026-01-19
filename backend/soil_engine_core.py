from PIL import Image
import numpy as np
import os
import glob

# ---------- VALIDATION ----------
def is_valid_soil_image(img):
    gray = np.array(img.convert("L").resize((100, 100))).astype(float)
    arr = np.array(img.resize((100, 100))).astype(float)

    std_gray = gray.std()
    mean_gray = gray.mean()
    mean_rgb = np.mean(arr, axis=(0, 1))

    # Reject white paper/notebook
    if mean_gray > 200 and std_gray < 15:
        return False
    # Reject very uniform images
    if std_gray < 5:
        return False
    # Reject pure white/black screenshots
    if np.all(mean_rgb > 240) or np.all(mean_rgb < 15):
        return False
    # Reject text-heavy images
    edges = np.abs(gray[1:, :] - gray[:-1, :])
    if edges.max() > 200 and edges.std() > 40:
        return False

    return True


# ---------- FEATURE EXTRACTION ----------
def extract_soil_features(img):
    arr = np.array(img.resize((150, 150))).astype(float)
    gray = np.array(img.convert("L").resize((150, 150))).astype(float)

    mean_rgb = np.mean(arr, axis=(0, 1))
    std_rgb = np.std(arr, axis=(0, 1))
    mean_gray = gray.mean()
    std_gray = gray.std()

    # Particle texture
    particle_texture = 0
    patch_count = 0
    for i in range(0, 140, 10):
        for j in range(0, 140, 10):
            patch = gray[i:i+10, j:j+10]
            particle_texture += patch.std()
            patch_count += 1
    particle_texture /= patch_count

    # Edge analysis
    edges_h = np.abs(gray[1:, :] - gray[:-1, :])
    edges_v = np.abs(gray[:, 1:] - gray[:, :-1])
    edge_mean = (edges_h.mean() + edges_v.mean()) / 2
    edge_std = (edges_h.std() + edges_v.std()) / 2

    color_uniformity = 1.0 / (std_gray + 1)

    hsv = np.array(Image.fromarray(arr.astype(np.uint8)).convert("HSV")).astype(float)
    mean_hsv = np.mean(hsv, axis=(0, 1))

    r, g, b = mean_rgb
    earth_tone = (r + g) / (b + 1)

    return {
        "mean_rgb": mean_rgb,
        "std_rgb": std_rgb,
        "mean_gray": mean_gray,
        "std_gray": std_gray,
        "particle_texture": particle_texture,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "color_uniformity": color_uniformity,
        "mean_hsv": mean_hsv,
        "earth_tone": earth_tone
    }


# ---------- DATASET LOADING ----------
def load_dataset():
    dataset = []
    base_paths = ["dataset", "./dataset", "../dataset"]
    dataset_path = None
    for path in base_paths:
        if os.path.exists(path) and os.path.isdir(path):
            dataset_path = path
            break
    if not dataset_path:
        return dataset

    for soil in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, soil)
        if not os.path.isdir(folder):
            continue

        for img_path in glob.glob(f"{folder}/*"):
            try:
                img = Image.open(img_path).convert("RGB")
                features = extract_soil_features(img)
                dataset.append({
                    "soil": soil.lower(),
                    "features": features
                })
            except:
                continue

    return dataset


DATASET = load_dataset()


# ---------- SIMILARITY CALCULATION ----------
def calculate_similarity(f1, f2):
    return (
        (f1["particle_texture"] - f2["particle_texture"])**2 * 0.30 +
        (f1["edge_mean"] - f2["edge_mean"])**2 * 0.20 +
        (f1["edge_std"] - f2["edge_std"])**2 * 0.15 +
        (f1["std_gray"] - f2["std_gray"])**2 * 0.12 +
        (f1["color_uniformity"] - f2["color_uniformity"])**2 * 0.10 +
        np.mean((f1["mean_rgb"] - f2["mean_rgb"])**2) * 0.08 +
        np.mean((f1["mean_hsv"] - f2["mean_hsv"])**2) * 0.05
    )


# ---------- FINAL CLASSIFIER ----------
def classify_soil(img):
    if not is_valid_soil_image(img):
        return {"error": True, "message": "Not a soil image"}

    if len(DATASET) == 0:
        return {"error": True, "message": "Dataset not loaded"}

    features = extract_soil_features(img)
    scores = []

    for item in DATASET:
        score = calculate_similarity(features, item["features"])
        scores.append((item["soil"], score))

    scores.sort(key=lambda x: x[1])

    votes = {}
    for i in range(min(25, len(scores))):
        soil = scores[i][0]
        votes[soil] = votes.get(soil, 0) + (25 - i)

    prediction = max(votes, key=votes.get)
    max_votes = sum(range(1, min(25, len(scores)) + 1))
    confidence = int((votes[prediction] / max_votes) * 100)

    # Apply soil-specific rules
    pt = features["particle_texture"]
    mg = features["mean_gray"]
    sg = features["std_gray"]
    em = features["edge_mean"]

    # Gravel
    if pt > 30 or em > 25 or sg > 45:
        gravel_votes = votes.get("gravel", 0)
        if gravel_votes > 50 or (pt > 35 and em > 20):
            prediction = "gravel"
    # Clay
    elif pt < 12 and sg < 20 and mg < 130:
        clay_votes = votes.get("clay", 0)
        if clay_votes > 40 or (pt < 10 and sg < 15):
            prediction = "clay"
    # Sand
    elif pt > 18 and mg > 140 and sg > 25:
        sand_votes = votes.get("sand", 0)
        if sand_votes > 40 or (pt > 22 and mg > 150):
            prediction = "sand"
    # Silt
    elif 12 <= pt <= 18 and 100 <= mg <= 140 and 20 <= sg <= 30:
        silt_votes = votes.get("silt", 0)
        if silt_votes > 40:
            prediction = "silt"

    return {
        "error": False,
        "soil": prediction,
        "confidence": confidence
    }
