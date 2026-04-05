import os

# 1. Download the 'Face Recognition' folder from your Google Drive
# 2. Place it inside your current folder: d:\UNiversityData\Semistet2\Machine Learning\MLProject\
# 3. If you name it 'Face Recognition', the path will be as follows:
data_path = './Face Recognition'  # local dataset path

if os.path.exists(data_path):
    print(f"Success! Found {len(os.listdir(data_path))} folders.")
else:
    print(f"Path not found: {data_path}. Please check that you downloaded and extracted the folder correctly.")

# Dataset Preparation for Face Verification
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# Filter persons with >= 20 images to speed up processing
min_images = 20
valid_people = []

if os.path.exists(data_path):
    for person in os.listdir(data_path):
        person_path = os.path.join(data_path, person)
        if not os.path.isdir(person_path):
            continue  # skip files (like saved embeddings)
        if len(os.listdir(person_path)) >= min_images:
            valid_people.append(person)
    
    print(f"Total persons with >= {min_images} images: {len(valid_people)}")
else:
    print("Please download the data first before running this cell.")

#use fewer pairs for faster processing
genuine_per_person = 5     #5 genuine pairs for each person
num_impostors = len(valid_people) * genuine_per_person  # impostor pairs

def create_balanced_pairs(data_path, people_list, genuine_per_person, impostor_pairs):
    genuine_pairs = []
    impostor_pairs_list = []

    #genuine pairs
    for person in people_list:
        imgs = os.listdir(os.path.join(data_path, person))
        for _ in range(genuine_per_person):
            img1, img2 = random.sample(imgs, 2)
            genuine_pairs.append((
                os.path.join(data_path, person, img1),
                os.path.join(data_path, person, img2),
                1
            ))

    #impostor pairs
    for _ in range(impostor_pairs):
        p1, p2 = random.sample(people_list, 2)
        img1 = random.choice(os.listdir(os.path.join(data_path, p1)))
        img2 = random.choice(os.listdir(os.path.join(data_path, p2)))
        impostor_pairs_list.append((
            os.path.join(data_path, p1, img1),
            os.path.join(data_path, p2, img2),
            0
        ))

    return genuine_pairs + impostor_pairs_list

pairs = create_balanced_pairs(data_path, valid_people, genuine_per_person, num_impostors)
print("Total pairs:", len(pairs))


%pip install insightface==0.7.3 onnxruntime tqdm opencv-python matplotlib scikit-learn


import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


#PyTorch backend, GPU if available
ctx_id = 0  #GPU id, -1 for CPU
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=ctx_id, det_size=(640,640))  #face detection size
print("ArcFace model loaded successfully!")


def get_arcface_embedding(img_path):
    """
    Returns 512-d normalized embedding for a single face.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


arcface_emb1 = []
arcface_emb2 = []
labels_clean = []  #for pairs where face detected

for p1, p2, label in tqdm(pairs):
    emb1 = get_arcface_embedding(p1)
    emb2 = get_arcface_embedding(p2)
    if emb1 is not None and emb2 is not None:
        arcface_emb1.append(emb1)
        arcface_emb2.append(emb2)
        labels_clean.append(label)

arcface_emb1 = np.array(arcface_emb1)
arcface_emb2 = np.array(arcface_emb2)
labels_clean = np.array(labels_clean)

print("Total valid pairs:", len(labels_clean))




import os
import numpy as np

# Paths to saved embeddings
emb1_path = '/content/drive/MyDrive/Face Recognition/arcface_emb1.npy'
emb2_path = '/content/drive/MyDrive/Face Recognition/arcface_emb2.npy'
labels_path = '/content/drive/MyDrive/Face Recognition/labels_clean.npy'

# Check if embeddings already exist
if os.path.exists(emb1_path) and os.path.exists(emb2_path) and os.path.exists(labels_path):
    arcface_emb1 = np.load(emb1_path)
    arcface_emb2 = np.load(emb2_path)
    labels_clean = np.load(labels_path)
    print(f"Embeddings loaded from disk! Total valid pairs: {len(labels_clean)}")
else:
    # Compute embeddings as before
    arcface_emb1 = []
    arcface_emb2 = []
    labels_clean = []

    for p1, p2, label in tqdm(pairs):
        emb1 = get_arcface_embedding(p1)
        emb2 = get_arcface_embedding(p2)
        if emb1 is not None and emb2 is not None:
            arcface_emb1.append(emb1)
            arcface_emb2.append(emb2)
            labels_clean.append(label)

    arcface_emb1 = np.array(arcface_emb1)
    arcface_emb2 = np.array(arcface_emb2)
    labels_clean = np.array(labels_clean)

    # Save for future use
    np.save(emb1_path, arcface_emb1)
    np.save(emb2_path, arcface_emb2)
    np.save(labels_path, labels_clean)
    print(f"Embeddings computed and saved! Total valid pairs: {len(labels_clean)}")




similarities = [cosine_similarity(e1, e2) for e1, e2 in zip(arcface_emb1, arcface_emb2)]
similarities = np.array(similarities)


def compute_far_frr(similarities, labels, threshold):
    preds = similarities >= threshold
    genuine = labels == 1
    impostor = labels == 0
    FAR = np.sum(preds[impostor] == 1) / np.sum(impostor)
    FRR = np.sum(preds[genuine] == 0) / np.sum(genuine)
    return FAR, FRR


thresholds = np.linspace(-1, 1, 200)
fars = []
frrs = []

for th in thresholds:
    far, frr = compute_far_frr(similarities, labels_clean, th)
    fars.append(far)
    frrs.append(frr)

# Equal Error Rate (EER)
eer_index = np.argmin(np.abs(np.array(fars) - np.array(frrs)))
eer = (fars[eer_index] + frrs[eer_index]) / 2
best_threshold = thresholds[eer_index]

print("Best Threshold:", best_threshold)
print("Equal Error Rate (EER):", eer)


# Add these two lines to print FAR and FRR at best threshold
far_at_best, frr_at_best = compute_far_frr(similarities, labels_clean, best_threshold)
print("FAR at best threshold:", far_at_best)
print("FRR at best threshold:", frr_at_best)


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,6))
plt.plot(thresholds, fars, label='FAR (False Acceptance Rate)', color='red')
plt.plot(thresholds, frrs, label='FRR (False Rejection Rate)', color='blue')
plt.axvline(best_threshold, color='green', linestyle='--', label=f'Best Threshold = {best_threshold:.4f}')
plt.xlabel("Threshold")
plt.ylabel("Error Rate")
plt.title("FAR and FRR vs Threshold")
plt.legend()
plt.grid(True)
plt.show()


fpr, tpr, _ = roc_curve(labels_clean, similarities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ArcFace ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate (FAR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - ArcFace")
plt.legend()
plt.grid(True)
plt.show()


import os

# Construct full paths using your data_path variable
img1_path = os.path.join(data_path, "Zico", "Zico_0001.jpg")
img2_path = os.path.join(data_path, "Nicole", "Nicole_0001.jpg")

# Optional: check if files exist
print(os.path.exists(img1_path), os.path.exists(img2_path))  # both should be True

# Run your embedding and verification
emb1 = get_arcface_embedding(img1_path)
emb2 = get_arcface_embedding(img2_path)

similarity = cosine_similarity(emb1, emb2)
print("Similarity:", similarity)
print("Verification result:", "Same person" if similarity >= best_threshold else "Different person")




!pip install facenet_pytorch

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("FaceNet model loaded on", device)


from torchvision import transforms

# Transform image to FaceNet input
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_facenet_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img_tensor)
        return emb.cpu().numpy().flatten()
    except Exception as e:
        print("Skipping pair", img_path, "due to error:", e)
        return None


genuine_per_person = 5
num_impostors = len(valid_people) * genuine_per_person

def create_balanced_pairs(data_path, people_list, genuine_per_person, impostor_pairs):
    genuine_pairs = []
    impostor_pairs_list = []

    #genuine pairs
    for person in people_list:
        imgs = os.listdir(os.path.join(data_path, person))
        for _ in range(genuine_per_person):
            img1, img2 = np.random.choice(imgs, 2, replace=False)
            genuine_pairs.append((
                os.path.join(data_path, person, img1),
                os.path.join(data_path, person, img2),
                1
            ))

    #impostor pairs
    for _ in range(impostor_pairs):
        p1, p2 = np.random.choice(people_list, 2, replace=False)
        img1 = np.random.choice(os.listdir(os.path.join(data_path, p1)))
        img2 = np.random.choice(os.listdir(os.path.join(data_path, p2)))
        impostor_pairs_list.append((
            os.path.join(data_path, p1, img1),
            os.path.join(data_path, p2, img2),
            0
        ))

    return genuine_pairs + impostor_pairs_list

pairs = create_balanced_pairs(data_path, valid_people, genuine_per_person, num_impostors)
print("Total pairs:", len(pairs))


facenet_emb1 = []
facenet_emb2 = []
labels_clean_facenet = []

for p1, p2, label in tqdm(pairs):
    emb1 = get_facenet_embedding(p1)
    emb2 = get_facenet_embedding(p2)
    if emb1 is not None and emb2 is not None:
        facenet_emb1.append(emb1)
        facenet_emb2.append(emb2)
        labels_clean_facenet.append(label)

facenet_emb1 = np.array(facenet_emb1)
facenet_emb2 = np.array(facenet_emb2)
labels_clean_facenet = np.array(labels_clean_facenet)

print("Total valid pairs for FaceNet:", len(labels_clean_facenet))




import os
import numpy as np
from tqdm import tqdm

# Paths to saved FaceNet embeddings
emb1_path = '/content/drive/MyDrive/Face Recognition/facenet_emb1.npy'
emb2_path = '/content/drive/MyDrive/Face Recognition/facenet_emb2.npy'
labels_path = '/content/drive/MyDrive/Face Recognition/labels_clean_facenet.npy'

# Check if embeddings already exist
if os.path.exists(emb1_path) and os.path.exists(emb2_path) and os.path.exists(labels_path):
    facenet_emb1 = np.load(emb1_path)
    facenet_emb2 = np.load(emb2_path)
    labels_clean_facenet = np.load(labels_path)
    print(f"FaceNet embeddings loaded from disk! Total valid pairs: {len(labels_clean_facenet)}")
else:
    # Compute embeddings as before
    facenet_emb1 = []
    facenet_emb2 = []
    labels_clean_facenet = []

    for p1, p2, label in tqdm(pairs):
        emb1 = get_facenet_embedding(p1)
        emb2 = get_facenet_embedding(p2)
        if emb1 is not None and emb2 is not None:
            facenet_emb1.append(emb1)
            facenet_emb2.append(emb2)
            labels_clean_facenet.append(label)

    facenet_emb1 = np.array(facenet_emb1)
    facenet_emb2 = np.array(facenet_emb2)
    labels_clean_facenet = np.array(labels_clean_facenet)

    # Save for future use
    np.save(emb1_path, facenet_emb1)
    np.save(emb2_path, facenet_emb2)
    np.save(labels_path, labels_clean_facenet)
    print(f"FaceNet embeddings computed and saved! Total valid pairs: {len(labels_clean_facenet)}")




#compute cosine similarities for all valid pairs
similarities_facenet = [cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0]
                        for e1, e2 in zip(facenet_emb1, facenet_emb2)]
similarities_facenet = np.array(similarities_facenet)


#function to compute FAR and FRR
def compute_far_frr(similarities, labels, threshold):
    preds = similarities >= threshold
    genuine = labels == 1
    impostor = labels == 0
    FAR = np.sum(preds[impostor] == 1) / np.sum(impostor)
    FRR = np.sum(preds[genuine] == 0) / np.sum(genuine)
    return FAR, FRR

#compute FAR and FRR over a range of thresholds
thresholds_facenet = np.linspace(-1, 1, 200)
fars_facenet = []
frrs_facenet = []

for th in thresholds_facenet:
    far, frr = compute_far_frr(similarities_facenet, labels_clean_facenet, th)
    fars_facenet.append(far)
    frrs_facenet.append(frr)

#Equal Error Rate (EER) and best threshold
eer_index = np.argmin(np.abs(np.array(fars_facenet) - np.array(frrs_facenet)))
eer_facenet = (fars_facenet[eer_index] + frrs_facenet[eer_index]) / 2
best_threshold_facenet = thresholds_facenet[eer_index]

print("FaceNet Best Threshold:", best_threshold_facenet)
print("FaceNet Equal Error Rate (EER):", eer_facenet)
print("FAR at best threshold:", fars_facenet[eer_index])
print("FRR at best threshold:", frrs_facenet[eer_index])

plt.figure(figsize=(8,6))
plt.plot(thresholds_facenet, fars_facenet, label='FAR (False Acceptance Rate)', color='red')
plt.plot(thresholds_facenet, frrs_facenet, label='FRR (False Rejection Rate)', color='blue')
plt.axvline(best_threshold_facenet, color='green', linestyle='--', label=f'Best Threshold = {best_threshold_facenet:.4f}')
plt.xlabel("Threshold")
plt.ylabel("Error Rate")
plt.title("FAR and FRR vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

#ROC Curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(labels_clean_facenet, similarities_facenet)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"FaceNet ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate (FAR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - FaceNet")
plt.legend()
plt.grid(True)
plt.show()


import os

# Construct full paths using your data_path variable
img1_path = os.path.join(data_path, "Zico", "Zico_0001.jpg")
img2_path = os.path.join(data_path, "Nicole", "Nicole_0001.jpg")

# Optional: check if files exist
print(os.path.exists(img1_path), os.path.exists(img2_path))  # both should be True

# Run your FaceNet embedding and verification
emb1 = get_facenet_embedding(img1_path)
emb2 = get_facenet_embedding(img2_path)

# Compute cosine similarity
similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
print("Similarity:", similarity)
print("Verification result:", "Same person" if similarity >= best_threshold_facenet else "Different person")




plt.figure(figsize=(8,6))

# ArcFace
plt.plot(thresholds, fars, label="ArcFace FAR", color="red")
plt.plot(thresholds, frrs, label="ArcFace FRR", color="blue")

# FaceNet
plt.plot(thresholds_facenet, fars_facenet, label="FaceNet FAR", color="yellow")
plt.plot(thresholds_facenet, frrs_facenet, label="FaceNet FRR", color="green")

# Optional: mark best thresholds
plt.axvline(best_threshold, color="brown", linestyle="--", label=f"ArcFace Best Threshold = {best_threshold:.4f}")
plt.axvline(best_threshold_facenet, color="green", linestyle="--", label=f"FaceNet Best Threshold = {best_threshold_facenet:.4f}")


plt.xlabel("Threshold")
plt.ylabel("Error Rate")
plt.title("FAR and FRR Comparison: ArcFace vs FaceNet")
plt.legend()
plt.grid(True)
plt.show()




!pip install --upgrade gradio




!pip install flask pyngrok pillow numpy


from pyngrok import ngrok

# Set your ngrok auth token
ngrok.set_auth_token("33mKNEbBWoK8QFofHPKpbjY67iU_uusaKH97bdQiaJqrCbt7")


from pyngrok import ngrok
ngrok.kill()  # Stop old tunnels




# Install dependencies
!pip install flask pyngrok pillow numpy opencv-python-headless --quiet

from flask import Flask, request, jsonify, send_file
from pyngrok import ngrok
from PIL import Image
import numpy as np
from threading import Thread

# Helper functions
def get_embedding(img):
    img = img.resize((160, 160))
    return np.array(img).flatten() / 255.0

def cosine_similarity(e1, e2):
    if e1 is None or e2 is None or np.linalg.norm(e1)==0 or np.linalg.norm(e2)==0:
        return 0.0
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

# Flask App
app = Flask(__name__)
reference_image = None

@app.route("/")
def index():
    return send_file("face_verification.html")

@app.route("/verify_upload", methods=["POST"])
def verify_upload():
    image1 = request.files['image1']
    image2 = request.files['image2']
    model = request.form.get("model", "ArcFace")
    img1 = Image.open(image1).convert("RGB")
    img2 = Image.open(image2).convert("RGB")
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    sim = cosine_similarity(emb1, emb2)
    status = "✅ Same person" if sim > 0.7 else "❌ Different person"
    return jsonify({"status": status, "similarity": round(sim,4), "model": model})

@app.route("/upload_reference", methods=["POST"])
def upload_reference():
    global reference_image
    img_file = request.files['reference']
    reference_image = Image.open(img_file).convert("RGB")
    return jsonify({"status": "Reference uploaded"})

@app.route("/verify_webcam", methods=["POST"])
def verify_webcam():
    global reference_image
    if reference_image is None:
        return jsonify({"status": "❌ Upload reference image first", "similarity": 0.0, "model": ""})
    webcam_image = request.files['webcam_image']
    model = request.form.get("model", "ArcFace")
    webcam_img = Image.open(webcam_image).convert("RGB")
    emb1 = get_embedding(reference_image)
    emb2 = get_embedding(webcam_img)
    sim = cosine_similarity(emb1, emb2)
    status = "✅ Same person" if sim > 0.7 else "❌ Different person"
    return jsonify({"status": status, "similarity": round(sim,4), "model": model})

# HTML GUI
html_code = """
<!DOCTYPE html>
<html>
<head>
<title>Face Verification App</title>
<style>
body { font-family: Arial; background-color: #fdf7ef; margin:0; padding:0; color:#013220; }
.header { background: linear-gradient(90deg, #ff8c00, #ffb347); padding:20px; color:white; text-align:center; font-size:28px; font-weight:bold; border-bottom:4px solid #013220; }
#tabs { text-align:center; margin-top:20px; }
#tabs button { background-color:#013220; color:white; padding:12px 25px; border:none; border-radius:6px; margin:5px; cursor:pointer; }
#tabs button:hover { background-color:#035c35; }
.tab { display:none; padding:20px; width:95%; margin:auto; background:#ffffff; border-radius:10px; box-shadow:0 0 10px #ccc; margin-top:20px; }
.tab.active { display:block; }
.section-title { font-size:22px; font-weight:bold; color:#ff8c00; margin-bottom:20px; text-align:center; }

.image-row { display:flex; justify-content:space-around; flex-wrap:wrap; margin-bottom:20px; }
.img-column { text-align:center; margin-bottom:15px; }
.img-column label { font-size:18px; font-weight:bold; display:inline-block; width:260px; }
.preview-box { width:260px; height:260px; border:3px solid #ff8c00; border-radius:12px; object-fit:cover; background:#fff3e0; margin-top:10px; }
select { padding:8px; border:2px solid #013220; border-radius:6px; font-size:16px; margin-left:10px; }
.green-btn { background-color:#32cd32; padding:12px 30px; color:white; border:none; border-radius:8px; font-size:18px; cursor:pointer; margin-top:20px; }
.green-btn:hover { background-color:#28b428; }
.result { text-align:center; font-size:20px; margin-top:15px; font-weight:bold; }
video { width:260px; height:260px; border:3px solid #013220; border-radius:12px; margin-top:10px; }

</style>
</head>
<body>

<div class="header">FACE VERIFICATION SYSTEM</div>

<div id="tabs">
    <button onclick="openTab('upload')">Image Verification</button>
    <button onclick="openTab('webcam')">Webcam Verification</button>
</div>

<!-- Image Verification -->
<div id="upload" class="tab active">
    <div class="section-title">Upload Images</div>
    <div class="image-row">
        <div class="img-column">
            <label>Reference Image</label><br>
            <input type="file" id="refImage" accept="image/*"><br>
            <img id="refPreview" class="preview-box">
        </div>
        <div class="img-column">
            <label>Verify Image</label><br>
            <input type="file" id="testImage" accept="image/*"><br>
            <img id="testPreview" class="preview-box">
        </div>
    </div>
    <div style="text-align:center; margin-top:20px;">
        <span style="font-size:18px; font-weight:bold;">Select Model:</span>
        <select id="modelSelect">
            <option value="ArcFace">ArcFace</option>
            <option value="FaceNet">FaceNet</option>
        </select>
    </div>
    <div style="text-align:center;">
        <button class="green-btn" id="verifyBtn">Verify</button>
    </div>
    <div class="result" id="resultText"></div>
</div>

<!-- Webcam Verification -->
<div id="webcam" class="tab">
    <div class="section-title">Webcam Verification</div>
    <div class="image-row">

        <!-- Column 1: Reference Image -->
        <div class="img-column">
            <label>Reference Image</label><br>
            <input type="file" id="refWebcamImage" accept="image/*"><br>
            <img id="refWebcamPreview" class="preview-box">
        </div>

        <!-- Column 2: Live Webcam + Capture button -->
        <div class="img-column">
            <label>Live Webcam</label><br>
            <video id="video" autoplay></video><br>
            <button class="green-btn" id="captureBtn">Capture</button>
        </div>

        <!-- Column 3: Captured Image + Verify button -->
        <div class="img-column">
            <label>Captured Image</label><br>
            <img id="capturedImage" class="preview-box"><br>
            <button class="green-btn" id="verifyCamBtn">Verify</button>
        </div>

    </div>

    <!-- Select Model -->
    <div style="text-align:center; margin-top:15px;">
        <span style="font-size:18px; font-weight:bold;">Select Model:</span>
        <select id="modelSelectWebcam">
            <option value="ArcFace">ArcFace</option>
            <option value="FaceNet">FaceNet</option>
        </select>
    </div>

    <!-- Result under model selection -->
    <div class="result" id="webcamResult"></div>

</div>

<script>
function openTab(tabName){
    document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
    document.getElementById(tabName).classList.add("active");
}

document.getElementById("refImage").onchange = e =>
    document.getElementById("refPreview").src = URL.createObjectURL(e.target.files[0]);
document.getElementById("testImage").onchange = e =>
    document.getElementById("testPreview").src = URL.createObjectURL(e.target.files[0]);
document.getElementById("refWebcamImage").onchange = e => {
    document.getElementById("refWebcamPreview").src = URL.createObjectURL(e.target.files[0]);
    let fd = new FormData();
    fd.append("reference", e.target.files[0]);
    fetch("/upload_reference",{method:"POST", body:fd});
};

document.getElementById("verifyBtn").onclick = async () => {
    let fd = new FormData();
    fd.append("image1", document.getElementById("refImage").files[0]);
    fd.append("image2", document.getElementById("testImage").files[0]);
    fd.append("model", document.getElementById("modelSelect").value);
    let r = await fetch("/verify_upload",{method:"POST", body:fd});
    let d = await r.json();
    document.getElementById("resultText").textContent =
        `Result: ${d.status} | Similarity: ${d.similarity} | Model: ${d.model}`;
};

let video = document.getElementById("video");
navigator.mediaDevices.getUserMedia({video:true}).then(s=>video.srcObject=s);

let capturedBlob = null;
document.getElementById("captureBtn").onclick = () => {
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video,0,0);
    canvas.toBlob(b=>{
        capturedBlob = b;
        document.getElementById("capturedImage").src = URL.createObjectURL(b);
    });
};

document.getElementById("verifyCamBtn").onclick = async () => {
    if(!capturedBlob){ alert("Capture an image first!"); return; }
    let fd = new FormData();
    fd.append("webcam_image", capturedBlob,"webcam.png");
    fd.append("model", document.getElementById("modelSelectWebcam").value);
    let r = await fetch("/verify_webcam",{method:"POST", body:fd});
    let d = await r.json();
    // Result now under model selection
    document.getElementById("webcamResult").textContent =
        `Result: ${d.status} | Similarity: ${d.similarity} | Model: ${d.model}`;
};
</script>
</body>
</html>
"""

# Write HTML file
with open("face_verification.html","w") as f:
    f.write(html_code)

# Run server
def run_app(): app.run()
Thread(target=run_app).start()

public_url = ngrok.connect(5000)
print("🔗 OPEN YOUR APP HERE:\n", public_url)


