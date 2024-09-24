import cv2
import numpy as np
import onnx
import onnxruntime

# Load the aligned face image
aligned_face_image = cv2.imread(
    "/egr/research-actionlab/shared/lyf_data/emotion/instruction_dataset/emotion_classification_dataset/dataset/AffectNet/AffectNet_v2/cropped_Annotated/2/0a38dfebfd242e0749a5cad36b4f680c09eaaaf79709d1e82721d047.jpg")

# Assuming the aligned face image is resized to 192x192
aligned_face_image_resized = cv2.resize(aligned_face_image, (192, 192))


# Convert the image to float32 and normalize
aligned_face_image_normalized = aligned_face_image_resized.astype(
    np.float32) / 255.0

# Transpose the image dimensions to match model input shape
aligned_face_image_transposed = np.transpose(
    aligned_face_image_normalized, (2, 0, 1))

# # Expand dimensions to match model input shape
aligned_face_image_input = np.expand_dims(
    aligned_face_image_transposed, axis=0)

# Load the ONNX model
onnx_model = onnx.load(
    "./landmark/2d106det.onnx")

# Get the input name of the model
input_name = onnx_model.graph.input[0].name

# Create an ONNX Runtime session
ort_session = onnxruntime.InferenceSession(
    "./landmark/2d106det.onnx")

# Perform inference
print(type(aligned_face_image_input))
output = ort_session.run(None, {input_name: aligned_face_image_input})

# Output will contain the model's predictions
# The embedding vector is the output of the layer before the last layer output
# embedding_vector = output[-2]
