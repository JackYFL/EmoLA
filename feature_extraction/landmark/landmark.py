import json
import cv2
import dlib
from multiprocessing import Pool
from tqdm import tqdm


def detect_landmarks(image_path):
    # Load the pre-trained facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "/home/liyifa11/MyCodes/EmoDQ/EmotionInstructData/shape_predictor_68_face_landmarks.dat")

    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Check if a face is detected
    if len(faces) == 0:
        return None

    # Determine the facial landmarks for the first face detected
    landmarks = predictor(gray, faces[0])

    # Extract landmark coordinates
    landmark_coords = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_coords.append((x, y))

    return landmark_coords


def process_image(json_data):
    image_path = json_data["image"]
    landmarks = detect_landmarks(image_path)
    if landmarks is not None:
        return {"image": image_path, "landmarks": landmarks}
    else:
        return None


def main():
    input_json_path = "/home/liyifa11/MyCodes/EmoDQ/EmotionInstructData/emo_instruction/after_ExpW_with_AffectNet_no_dup.json"
    output_json_path = "/home/liyifa11/MyCodes/EmoDQ/EmotionInstructData/feature_extraction/landmark.json"

    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Use multiprocessing to parallelize the processing
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_image, data), total=len(data)))

    # Filter out None results (images without detected faces)
    results = [r for r in results if r is not None]

    # Write the results to a new JSON file
    with open(output_json_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
