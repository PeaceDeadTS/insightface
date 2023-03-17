import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
import json
from tqdm import tqdm
import sys
sys.path.append("/mnt/f/server/nn/insightface/detection/retinaface")
from retinaface import RetinaFace
import imgaug.augmenters as iaa

model_path = "/mnt/f/server/nn/insightface/models/model.onnx"
sess_options = ort.SessionOptions()
sess = ort.InferenceSession(model_path, sess_options)
input_name = sess.get_inputs()[0].name
retinaface_detector = RetinaFace("/mnt/f/server/nn/insightface/models/retinaface/R50", 0, ctx_id=-1)

def preprocess(image, image_size=(112, 112)):
    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    image = (image - 127.5) / 128.0
    return image

def extract_embedding(image):
    input_tensor = preprocess(image)
    embeddings = sess.run(None, {input_name: input_tensor})
    return embeddings[0]

def detect_and_align_face(image):
    bbox, landmarks = retinaface_detector.detect(image, threshold=0.8)
    if bbox is not None:
        aligned_face = retinaface_detector.align(image, landmarks[0])
        return aligned_face
    return None

def augment_image(image):
    seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-20, 20), scale=(0.8, 1.2))
    ])
    return seq(image=image)

def cosine_similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def save_embeddings(embeddings_dict, file_path="/mnt/f/server/nn/embeddings.json"):
    with open(file_path, "w") as f:
        json.dump(embeddings_dict, f)

def load_embeddings(file_path="/mnt/f/server/nn/embeddings.json"):
    with open(file_path, "r") as f:
        embeddings_dict = json.load(f)
    return embeddings_dict

test_image_path = "/mnt/f/server/nn/image.jpg"
test_image = cv2.imread(test_image_path)
test_image = detect_and_align_face(test_image)
test_embedding = extract_embedding(test_image)

image_folder = "/mnt/f/server/nn/images"
embeddings_file = "/mnt/f/server/nn/embeddings.json"

if not os.path.exists(embeddings_file):
    embeddings_dict = {}
    for image_path in tqdm(glob.glob(os.path.join(image_folder, "*.jpg"))):
        image = cv2.imread(image_path)
        aligned_face = detect_and_align_face(image)
        if aligned_face is None:
            continue
        augmented_faces = [aligned_face, augment_image(aligned_face)]

        for face in augmented_faces:
            embedding = extract_embedding(face)
            embeddings_dict[image_path] = embedding.tolist()

    save_embeddings(embeddings_dict, embeddings_file)
else:
    embeddings_dict = load_embeddings(embeddings_file)

threshold = 0.5
lookalikes = []

for image_path, embedding in tqdm(embeddings_dict.items()):
    similarity = cosine_similarity(test_embedding, np.array(embedding))

    if similarity > threshold:
        lookalikes.append((image_path, similarity))

if lookalikes:
    print("Lookalikes found:")
    lookalikes.sort(key=lambda x: x[1], reverse=True)
    for image_path, similarity in lookalikes:
        print(f"Image path: {image_path}, similarity: {similarity}")
else:
    print("No lookalikes found.")