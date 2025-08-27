from pathlib import Path
import pickle
from collections import Counter
import cv2
import face_recognition
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import os
from utils import SecurityUtils

class Core:
    def __init__(self):
        load_dotenv()
        self.DEFAULT_ENCODINGS_PATH = Path(
            os.getenv("ENCODINGS_PATH", "output/encoding.pkl")
        )
        self.BOUNDING_BOX_KNOWN_COLOR = os.getenv("BOUNDING_BOX_KNOWN_COLOR")
        self.BOUNDING_BOX_UNKNOWN_COLOR = os.getenv("BOUNDING_BOX_UNKNOWN_COLOR")


    # =====================
    # Train
    # =====================
    def encode_known_faces(self, model: str = "hog", encodings_location: Path = None) -> None:
        if encodings_location is None:
            encodings_location = self.DEFAULT_ENCODINGS_PATH

        names: list[str] = []
        encodings: list = []

        for filepath in Path("training").glob("*/*"):
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image, model=model)
            face_encs = face_recognition.face_encodings(image, face_locations)
            for enc in face_encs:
                names.append(name)
                encodings.append(enc)

        payload = {"names": names, "encodings": encodings}
        encodings_location.parent.mkdir(parents=True, exist_ok=True)
        with encodings_location.open("wb") as f:
            pickle.dump(payload, f)

        print(f"✅ Đã lưu encodings: {encodings_location} "
              f"(persons={len(set(names))}, faces={len(encodings)})")

    # =====================
    # Nhận diện từ ảnh tĩnh (vẽ bằng PIL)
    # =====================
    def recognize_faces_from_image(self, image_location: str, model: str = "hog", encodings_location: Path = None) -> None:
        if encodings_location is None:
            encodings_location = self.DEFAULT_ENCODINGS_PATH

        with encodings_location.open("rb") as f:
            loaded = pickle.load(f)

        input_image = face_recognition.load_image_file(image_location)
        input_face_locations = face_recognition.face_locations(input_image, model=model)
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        pillow_image = Image.fromarray(input_image)
        draw = ImageDraw.Draw(pillow_image)

        for bbox, unknown_enc in zip(input_face_locations, input_face_encodings):
            name = self._recognize_face(unknown_enc, loaded)
            if not name:
                name = "Unknown"
                color = self.BOUNDING_BOX_UNKNOWN_COLOR
            else:
                color = self.BOUNDING_BOX_KNOWN_COLOR
            self._display_face_pil(draw, bbox, name, color)

        del draw
        pillow_image.show()

    # =====================
    # Nhận diện từ webcam (OpenCV)
    # =====================
    def recognize_from_webcam(self, model: str = "hog", encodings_location: Path = None):
        if encodings_location is None:
            encodings_location = self.DEFAULT_ENCODINGS_PATH

        with encodings_location.open("rb") as f:
            loaded = pickle.load(f)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Webcam] Không mở được camera")
            return

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb, model=model)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            for bbox, face_enc in zip(face_locations, face_encodings):
                name = self._recognize_face(face_enc, loaded)
                if not name:
                    name = "Unknown"
                    color = self.BOUNDING_BOX_UNKNOWN_COLOR
                    SecurityUtils.alert_intruder(frame, bbox)
                else:
                    color = self.BOUNDING_BOX_KNOWN_COLOR

                top, right, bottom, left = bbox
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 2, bottom - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Face Recognition (press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # =====================
    # Helpers
    # =====================
    def _recognize_face(self, unknown_encoding, loaded_encodings: dict):
        boolean_matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding
        )
        votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
        if votes:
            return votes.most_common(1)[0][0]
        return None

    def _display_face_pil(self, draw: ImageDraw.ImageDraw, bounding_box, name: str, color: str):
        top, right, bottom, left = bounding_box
        draw.rectangle(((left, top), (right, bottom)), outline=color, width=2)
        try:
            text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
        except Exception:
            text_left, text_top, text_right, text_bottom = left, bottom, left + 8 * len(name) + 6, bottom + 18
        draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=color, outline=color)
        draw.text((text_left + 2, text_top + 1), name, fill="white")

    # =====================
    # Validate
    # =====================
    def validate(self, model: str = "hog", encodings_location: Path = None):
        if encodings_location is None:
            encodings_location = self.DEFAULT_ENCODINGS_PATH

        for filepath in Path("validation").rglob("*"):
            if filepath.is_file():
                self.recognize_faces_from_image(
                    image_location=str(filepath.absolute()),
                    model=model,
                    encodings_location=encodings_location,
                )
