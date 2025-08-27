from pathlib import Path
import pickle
from datetime import datetime
import cv2
import face_recognition
from core import Core
from auth import Auth
from dotenv import load_dotenv
import os


class FaceAdd:
    def __init__(self):
        load_dotenv()
        self.DEFAULT_ENCODINGS_PATH = Path(
            os.getenv("ENCODINGS_PATH", "output/encoding.pkl")
        )
        self.core = Core() 
    def capture_images_for_name(self, name: str, count: int = 10) -> list[str]:
        """Mở webcam, hướng dẫn người dùng thay đổi góc mặt. Nhấn 'c' để chụp, 'q' để hủy."""
        prompts = [
            "Center - look straight",
            "Turn slightly left",
            "Turn slightly right",
            "Look up",
            "Look down",
            "Smile / different expression",
            "Turn left more",
            "Turn right more",
            "Tilt head left",
            "Tilt head right",
        ]
        save_dir = Path("training") / name
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không mở được camera")
            return []

        captured_paths = []
        idx = 0
        print("👉 Hướng dẫn: đặt khuôn mặt vào khung. Khi sẵn sàng, nhấn 'c' để chụp, 'q' để dừng.")
        while idx < count:
            ok, frame = cap.read()
            if not ok:
                break
            prompt = prompts[idx] if idx < len(prompts) else f"Pose #{idx+1}"
            disp = frame.copy()
            cv2.putText(disp, f"Pose: {prompt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(disp, f"Captured: {idx}/{count} - press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow("Capture for training - Press c to capture", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # lưu ảnh
                filename = save_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(str(filename), frame)
                captured_paths.append(str(filename))
                idx += 1
                print(f"✅ Captured {idx}/{count}: {filename}")
            elif key == ord('q'):
                print("❌ User canceled capture.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return captured_paths

    def add_face_via_webcam_flow(self, model) -> None:
        """Flow thêm khuôn mặt mới: login root -> nhập tên -> capture 10 ảnh -> train -> validate"""
        # 1) kiểm tra root
        if not Path("auth/root.hash").exists():
            print("⚠️ Chưa có root account. Tạo root trước.")
            Auth.create_root_account()
        if not Auth.verify_root_login():
            print("❌ Đăng nhập thất bại. Hủy.")
            return

        # 2) nhập tên
        name = input("Nhập tên người cần thêm (không dấu cách): ").strip()
        if not name:
            print("⚠️ Tên không hợp lệ.")
            return

        # 3) capture
        captured = self.capture_images_for_name(name, count=10)
        if len(captured) < 10:
            print("⚠️ Số ảnh chụp không đủ. Xóa ảnh tạm (nếu có) và thử lại.")
            return

        # 4) train lại toàn bộ
        print("🔄 Bắt đầu huấn luyện (encode)...")
        self.core.encode_known_faces(model=model, encodings_location=self.DEFAULT_ENCODINGS_PATH)

        # 5) validate các ảnh mới
        print("🔎 Kiểm tra lại 10 ảnh vừa chụp...")
        with self.DEFAULT_ENCODINGS_PATH.open("rb") as f:
            loaded = pickle.load(f)

        bad_count = 0
        for img_path in captured:
            img = face_recognition.load_image_file(img_path)
            locs = face_recognition.face_locations(img, model=model)
            encs = face_recognition.face_encodings(img, locs)
            recognized_names = [self.core._recognize_face(e, loaded) for e in encs]

            if not recognized_names or any(r != name for r in recognized_names):
                bad_count += 1
                print(f"❌ Validation fail: {img_path} -> {recognized_names}")
            else:
                print(f"✅ Validation ok: {img_path} -> {recognized_names}")

        if bad_count == 0:
            print(f"🎉 Thêm khuôn mặt '{name}' thành công! Tất cả ảnh đều nhận dạng chính xác.")
        else:
            print(f"⚠️ Có {bad_count} ảnh không pass validation. Vui lòng xem lại hoặc chụp lại.")
