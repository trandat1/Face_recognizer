from pathlib import Path
import pickle
from collections import Counter
import cv2
import numpy as np
import insightface
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import os
from utils import SecurityUtils


class Core:
    def __init__(self):
        load_dotenv()
        # giữ tên file pkl giống code cũ để tránh nhầm lẫn
        self.DEFAULT_ENCODINGS_PATH = Path(os.getenv("ENCODINGS_PATH", "output/encodings.pkl"))

        # đảm bảo các thư mục tồn tại
        Path("training").mkdir(parents=True, exist_ok=True)
        Path("output").mkdir(parents=True, exist_ok=True)
        Path("validation").mkdir(parents=True, exist_ok=True)
        self.DEFAULT_ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

        # màu viền (mặc định nếu không đặt .env)
        def _parse_color(s, default):
            try:
                return tuple(map(int, s.split(",")))
            except Exception:
                return default

        self.BOUNDING_BOX_KNOWN_COLOR = _parse_color(
            os.getenv("BOUNDING_BOX_KNOWN_COLOR", "0,255,0"), (0, 255, 0)
        )
        self.BOUNDING_BOX_UNKNOWN_COLOR = _parse_color(
            os.getenv("BOUNDING_BOX_UNKNOWN_COLOR", "0,0,255"), (0, 0, 255)
        )

        # khởi tạo InsightFace
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1)

        # nếu file encodings chưa tồn tại, tạo file rỗng (tương thích format cũ & mới)
        if not self.DEFAULT_ENCODINGS_PATH.exists():
            empty_data = {
                "names": [],
                # embeddings là numpy array (shape (0,512))
                "embeddings": np.zeros((0, 512), dtype=np.float32),
                "filepaths": [],
            }
            with open(self.DEFAULT_ENCODINGS_PATH, "wb") as f:
                pickle.dump(empty_data, f)
            print(f"[INIT] Tạo file encodings rỗng tại: {self.DEFAULT_ENCODINGS_PATH}")

    # helper: load encodings và chuyển sang dạng chuẩn {'names','embeddings','filepaths'}
    def _load_encodings(self, encodings_location: Path):
        with open(encodings_location, "rb") as f:
            data = pickle.load(f)
        # tương thích với file cũ của face_recognition (key 'encodings')
        if "embeddings" in data:
            embeddings = np.array(data["embeddings"])
        elif "encodings" in data:
            embeddings = np.array(data["encodings"])
        else:
            embeddings = np.zeros((0, 512), dtype=np.float32)
        return {
            "names": data.get("names", []),
            "embeddings": embeddings,
            "filepaths": data.get("filepaths", []),
        }

    # =====================
    # Train (encode)
    # =====================
    def encode_known_faces(self, encodings_location: Path = None) -> None:
        if encodings_location is None:
            encodings_location = self.DEFAULT_ENCODINGS_PATH

        names, embeddings, filepaths = [], [], []

        for filepath in Path("training").glob("*/*"):
            name = filepath.parent.name
            img = cv2.imread(str(filepath))
            if img is None:
                print(f"[WARN] Không đọc được ảnh: {filepath}")
                continue

            faces = self.app.get(img)
            if not faces:
                print(f"[WARN] Không phát hiện khuôn mặt trong: {filepath}")
                continue

            for face in faces:
                names.append(name)
                embeddings.append(face.normed_embedding.astype(np.float32))
                filepaths.append(str(filepath))

        # nếu không tìm được embedding nào, vẫn tạo file (array shape (0,512))
        if len(embeddings) > 0:
            emb_array = np.vstack(embeddings)
        else:
            emb_array = np.zeros((0, 512), dtype=np.float32)

        data = {"names": names, "embeddings": emb_array, "filepaths": filepaths}
        encodings_location.parent.mkdir(parents=True, exist_ok=True)
        with open(encodings_location, "wb") as f:
            pickle.dump(data, f)

        print(f"✅ Đã lưu {len(names)} embeddings vào {encodings_location}")

    # =====================
    # Nhận diện từ ảnh tĩnh (PIL)
    # =====================
    def recognize_faces_from_image(
        self,
        image_location: str,
        encodings_location: Path = None,
        threshold: float = 0.46,
        show_image: bool = True,
        save_result: bool = True,
        out_path: str = "output/result.jpg",
    ) -> None:
        if encodings_location is None:
            encodings_location = self.DEFAULT_ENCODINGS_PATH

        loaded = self._load_encodings(encodings_location)
        if loaded["embeddings"].size == 0:
            print("[WARN] Không có embeddings trong file. Hãy chạy encode_known_faces() trước.")
            # tạo ảnh và trả về (hoặc vẫn hiển thị gốc)
        img = cv2.imread(image_location)
        if img is None:
            print(f"[ERR] Không đọc được ảnh: {image_location}")
            return

        faces = self.app.get(img)
        print(f"Số khuôn mặt phát hiện: {len(faces)}")

        pillow_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pillow_image)
        font = ImageFont.load_default()

        for i, face in enumerate(faces):
            emb = face.normed_embedding
            sims = np.dot(loaded["embeddings"], emb) if loaded["embeddings"].size > 0 else np.array([])
            if sims.size == 0:
                best_idx = -1
                best_score = 0.0
            else:
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])

            if best_score >= threshold:
                name = loaded["names"][best_idx]
                color = self.BOUNDING_BOX_KNOWN_COLOR
            else:
                name = "Unknown"
                color = self.BOUNDING_BOX_UNKNOWN_COLOR

            print(f"[Face {i}] => {name}, similarity={best_score:.4f}", end="")
            if best_idx >= 0 and loaded.get("filepaths"):
                print(f", file={loaded['filepaths'][best_idx]}")
            else:
                print("")

            # face.bbox is [x1, y1, x2, y2]
            x1, y1, x2, y2 = face.bbox.astype(int)
            # dùng PIL để vẽ
            self._display_face_pil(draw, (x1, y1, x2, y2), name, color, font=font)

        # lưu và/hoặc show
        if save_result:
            pillow_image.convert("RGB").save(out_path)
            print(f"Ảnh kết quả đã lưu tại: {out_path}")

        if show_image:
            pillow_image.show()

    # =====================
    # Nhận diện từ webcam (OpenCV)
    # =====================
    def recognize_from_webcam(self, encodings_location: Path = None, threshold: float = 0.46):
        if encodings_location is None:
            encodings_location = self.DEFAULT_ENCODINGS_PATH

        loaded = self._load_encodings(encodings_location)
        if loaded["embeddings"].size == 0:
            print("[WARN] Không có embeddings trong file. Hãy chạy encode_known_faces() trước.")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[Webcam] Không mở được camera")
            return

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            faces = self.app.get(frame)

            for face in faces:
                emb = face.normed_embedding
                sims = np.dot(loaded["embeddings"], emb) if loaded["embeddings"].size > 0 else np.array([])
                if sims.size == 0:
                    best_idx = -1
                    best_score = 0.0
                else:
                    best_idx = int(np.argmax(sims))
                    best_score = float(sims[best_idx])

                if best_score >= threshold:
                    name = loaded["names"][best_idx]
                    color = self.BOUNDING_BOX_KNOWN_COLOR
                else:
                    name = "Unknown"
                    color = self.BOUNDING_BOX_UNKNOWN_COLOR
                    # gọi SecurityUtils với bbox ở dạng (top, right, bottom, left) giống face_recognition
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    bbox_for_security = (y1, x2, y2, x1)
                    try:
                        SecurityUtils.alert_intruder(frame, bbox_for_security)
                    except Exception as e:
                        # không để crash nếu function alert có kỳ vọng khác
                        print(f"[WARN] alert_intruder lỗi: {e}")

                # vẽ bounding box lên frame (OpenCV expects BGR)
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

            cv2.imshow("Face Recognition (press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # =====================
    # Helpers
    # =====================
    def _display_face_pil(self, draw: ImageDraw.ImageDraw, bounding_box, name: str, color: str, font=None):
        """
        Vẽ khung và tên bằng PIL, tương thích Pillow mới (>=10).
        """
        x1, y1, x2, y2 = bounding_box
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)

        if font is None:
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
            except Exception:
                font = None

        # Tính kích thước text
        try:
            if hasattr(draw, "textbbox"):  # Pillow >= 8
                text_left, text_top, text_right, text_bottom = draw.textbbox((x1, y2), name, font=font)
            else:  # Fallback cho Pillow cũ
                w, h = draw.textsize(name, font=font)
                text_left, text_top = x1, y2
                text_right, text_bottom = x1 + w, y2 + h
        except Exception:
            # fallback an toàn
            text_left, text_top, text_right, text_bottom = x1, y2, x1 + 8 * len(name) + 6, y2 + 18

        # Vẽ nền và chữ
        draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=color, outline=color)
        draw.text((text_left + 2, text_top + 1), name, fill="white", font=font)

    # =====================
    # Validate
    # =====================
    def validate(self, encodings_location: Path = None, threshold: float = 0.46):
        if encodings_location is None:
            encodings_location = self.DEFAULT_ENCODINGS_PATH

        for filepath in Path("validation").rglob("*"):
            if filepath.is_file():
                self.recognize_faces_from_image(
                    image_location=str(filepath.absolute()),
                    encodings_location=encodings_location,
                    threshold=threshold,
                )
