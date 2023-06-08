import cv2
import numpy as np
from PIL import Image

from milvus import Milvus, IndexType, MetricType
from pymilvus import connections, utility
from pymilvus_orm import Collection, FieldSchema, CollectionSchema

# Connect to Milvus server
connections.connect(host='localhost', port='19530')

# Define the collection name and dimension for face embeddings
collection_name = 'face_collection'
dimension = 512

# Load the pre-trained face recognition model (e.g., using OpenCV or any other library)
face_recognition_model = cv2.dnn.readNetFromTorch('face_recognition_model.t7')

# Define a function to extract face embeddings from an image
def extract_face_embeddings(image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (150, 150), (104.0, 177.0, 123.0), False, False)
    face_recognition_model.setInput(blob)
    return face_recognition_model.forward()

# Load the face images and extract embeddings
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Replace with actual image paths
face_embeddings = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_embeddings.append(extract_face_embeddings(rgb_image)[0])

# Convert face embeddings to numpy array
face_embeddings = np.array(face_embeddings)

# Create a Milvus collection
collection = Collection(name=collection_name)

# Check if the collection already exists
if utility.has_collection(collection_name):
    collection.load()
else:
    # Define the schema of the collection
    field = FieldSchema(name='embedding', dtype='float_vector', dim=dimension)
    schema = CollectionSchema(fields=[field], description='Face embeddings collection')
    collection.create_schema(schema)

# Insert face embeddings into the collection
collection.insert([face_embeddings.tolist()])

# Create an index for fast similarity search
index_param = {'index_type': IndexType.IVF_SQ8, 'metric_type': MetricType.L2, 'params': {'nlist': 128}}
collection.create_index('embedding', index_param)

# Perform a face recognition search
query_image_path = 'query_image.jpg'  # Replace with the path to the query image
query_image = cv2.imread(query_image_path)
query_rgb_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
query_embedding = extract_face_embeddings(query_rgb_image)[0]

# Convert the query embedding to a numpy array
query_embedding = np.array([query_embedding.tolist()])

# Perform similarity search
search_param = {'nprobe': 16}
results = collection.search(query_records=query_embedding, top_k=5, params=search_param)

# Retrieve the similar face images
similar_image_paths = [image_paths[result.id] for result in results[0]]

# Display the similar face images
for image_path in similar_image_paths:
    similar_image = Image.open(image_path)
    similar_image.show()
