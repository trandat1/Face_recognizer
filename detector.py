from pathlib import Path
import argparse
import os
import pickle
from collections import Counter
from datetime import datetime, timedelta

import cv2
import face_recognition
from PIL import Image, ImageDraw

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

try:
    # Tùy chọn: hỗ trợ .env nếu người dùng cài python-dotenv
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# =====================
# Cấu hình & hằng số
# =====================
DEFAULT_ENCODINGS_PATH = Path("output/encoding.pkl")
BOUNDING_BOX_KNOWN_COLOR_BGR = (255, 0, 0)   # Xanh (BGR) cho người đã train
BOUNDING_BOX_UNKNOWN_COLOR_BGR = (0, 0, 255) # Đỏ (BGR) cho Unknown
BOUNDING_BOX_KNOWN_COLOR_PIL = "blue"       # Cho ảnh tĩnh (PIL)
BOUNDING_BOX_UNKNOWN_COLOR_PIL = "red"

# Email qua biến môi trường (.env nếu có):
ALERT_SENDER = os.getenv("ALERT_SENDER", "your_email@gmail.com")
ALERT_PASSWORD = os.getenv("ALERT_PASSWORD", "")  # App Password nếu dùng Gmail
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", ALERT_SENDER)
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "300"))  # 5 phút mặc định
LAST_EMAIL_MARKER = Path("logs/last_email.txt")

# =====================
# CLI
# =====================
parser = argparse.ArgumentParser(description="Recognize faces from images or webcam")
parser.add_argument("--train", action="store_true", help="Train on input data in ./training/")
parser.add_argument("--validate", action="store_true", help="Validate using images in ./validation/")
parser.add_argument("--test", action="store_true", help="Test the model with a single image (-f)")
parser.add_argument("-f", dest="test_image", action="store", help="Path to an image with an unknown/known face")
parser.add_argument("--webcam", action="store_true", help="Run face recognition with webcam")
parser.add_argument("-m", dest="model", action="store", default="hog", choices=["hog", "cnn"], help="Model for detection: hog (CPU) | cnn (GPU)")
args = parser.parse_args()

# =====================
# Khởi tạo thư mục
# =====================
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)
Path("intruder").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# =====================
# Helper: Gửi email có đính kèm, có giới hạn tần suất
# =====================

def _can_send_email_now() -> bool:
    if not ALERT_PASSWORD or not ALERT_SENDER:
        # Nếu chưa cấu hình thì coi như không gửi được
        return False
    if not LAST_EMAIL_MARKER.exists():
        return True
    try:
        ts = LAST_EMAIL_MARKER.read_text().strip()
        last_time = datetime.fromisoformat(ts)
    except Exception:
        return True
    return datetime.now() - last_time >= timedelta(seconds=ALERT_COOLDOWN_SECS)


def _touch_email_marker() -> None:
    LAST_EMAIL_MARKER.write_text(datetime.now().isoformat())


def send_alert_email(image_path: str, receiver: str | None = None) -> None:
    """Gửi email cảnh báo kèm ảnh Unknown. Tự động tôn trọng cooldown để tránh spam."""
    if not _can_send_email_now():
        print("[Email] Đang trong thời gian cooldown, bỏ qua gửi email.")
        return

    sender = ALERT_SENDER
    password = ALERT_PASSWORD
    to_addr = receiver or ALERT_RECEIVER

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = to_addr
        msg["Subject"] = "⚠️ Cảnh báo an ninh: Phát hiện khuôn mặt lạ"
        body = "Hệ thống phát hiện khuôn mặt chưa được nhận diện. Vui lòng kiểm tra ảnh đính kèm."
        msg.attach(MIMEText(body, "plain"))

        with open(image_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={Path(image_path).name}")
        msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)

        _touch_email_marker()
        print(f"📧 Email cảnh báo đã gửi kèm ảnh: {image_path}")
    except Exception as e:
        print(f"[Email] Lỗi gửi email: {e}")


# =====================
# Train: sinh encodings
# =====================

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
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
    with encodings_location.open("wb") as f:
        pickle.dump(payload, f)
    print(f"✅ Đã lưu encodings: {encodings_location} (persons={len(set(names))}, faces={len(encodings)})")


# =====================
# Nhận diện từ ảnh tĩnh (vẽ bằng PIL)
# =====================

def recognize_faces_from_image(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    with encodings_location.open("rb") as f:
        loaded = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bbox, unknown_enc in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_enc, loaded)
        if not name:
            name = "Unknown"
            color = BOUNDING_BOX_UNKNOWN_COLOR_PIL
        else:
            color = BOUNDING_BOX_KNOWN_COLOR_PIL
        _display_face_pil(draw, bbox, name, color)

    del draw
    pillow_image.show()


# =====================
# Nhận diện từ webcam (vẽ bằng OpenCV), + log + lưu ảnh + email kèm ảnh
# =====================

def recognize_from_webcam(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH):
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
            name = _recognize_face(face_enc, loaded)
            if not name:
                name = "Unknown"
                color = BOUNDING_BOX_UNKNOWN_COLOR_BGR
                log_intrusion(name)
                saved_path = save_intruder_image(frame, bbox)
                # Gửi email kèm ảnh (tôn trọng cooldown)
                send_alert_email(saved_path)
            else:
                color = BOUNDING_BOX_KNOWN_COLOR_BGR

            top, right, bottom, left = bbox
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 2, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Recognition (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================
# Core helpers
# =====================

def _recognize_face(unknown_encoding, loaded_encodings: dict):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]
    return None


def _display_face_pil(draw: ImageDraw.ImageDraw, bounding_box, name: str, color: str):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=color, width=2)
    # Đo kích thước text và vẽ nền phía dưới
    try:
        text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    except Exception:
        # Fallback với PIL cũ: ước lượng box đơn giản
        text_left, text_top, text_right, text_bottom = left, bottom, left + 8 * len(name) + 6, bottom + 18
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=color, outline=color)
    draw.text((text_left + 2, text_top + 1), name, fill="white")


# =====================
# Logging & lưu ảnh Unknown
# =====================

def log_intrusion(name: str = "Unknown"):
    logline = f"[{datetime.now().isoformat()}] Detected: {name}\n"
    with open("logs/intrusion.log", "a", encoding="utf-8") as f:
        f.write(logline)
    print(logline.strip())


def save_intruder_image(frame, bounding_box) -> str:
    top, right, bottom, left = bounding_box
    # Thêm margin để crop dễ nhìn hơn
    margin = 20
    h, w = frame.shape[:2]
    t = max(0, top - margin)
    l = max(0, left - margin)
    b = min(h, bottom + margin)
    r = min(w, right + margin)
    face_image = frame[t:b, l:r]

    filename = Path("intruder") / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    cv2.imwrite(str(filename), face_image)
    print(f"[Saved] Ảnh Unknown: {filename}")
    return str(filename)


# =====================
# Validate
# =====================

def validate(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces_from_image(image_location=str(filepath.absolute()), model=model, encodings_location=encodings_location)


# =====================
# Main
# =====================
if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.model)
    if args.validate:
        validate(model=args.model)
    if args.test:
        if not args.test_image:
            raise SystemExit("--test cần kèm đường dẫn ảnh bằng -f")
        recognize_faces_from_image(image_location=args.test_image, model=args.model)
    if args.webcam:
        recognize_from_webcam(model=args.model)

# =====================
# Additional admin / capture helpers
# =====================

def _hash_password(password: str, salt: str) -> str:
    import hashlib
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def create_root_account() -> None:
    """Tạo tài khoản root (lưu hash vào file 'auth/root.hash')."""
    from getpass import getpass
    auth_dir = Path("auth")
    auth_dir.mkdir(exist_ok=True)
    root_file = auth_dir / "root.hash"
    if root_file.exists():
        print("Root account đã tồn tại. Nếu muốn tạo mới, hãy xóa 'auth/root.hash' trước.")
        return
    salt = datetime.now().strftime("%Y%m%d%H%M%S")
    pwd1 = getpass("Nhập mật khẩu root mới: ")
    pwd2 = getpass("Nhập lại mật khẩu: ")
    if pwd1 != pwd2:
        print("Mật khẩu không khớp. Hủy.")
        return
    hashed = _hash_password(pwd1, salt)
    # lưu dưới dạng salt$hash
    root_file.write_text(salt + "$" + hashed)
    print("✅ Tạo root account thành công.")


def verify_root_login() -> bool:
    """Yêu cầu nhập mật khẩu root và kiểm tra. Trả về True nếu đúng."""
    from getpass import getpass
    root_file = Path("auth/root.hash")
    if not root_file.exists():
        print("Chưa có root account. Bạn cần tạo bằng cách chạy create_root_account() hoặc dùng --create-root flag.")
        return False
    content = root_file.read_text().strip()
    if "$" not in content:
        print("Dữ liệu root bị hỏng.")
        return False
    salt, saved_hash = content.split("$", 1)
    pwd = getpass("Nhập mật khẩu root: ")
    return _hash_password(pwd, salt) == saved_hash


def capture_images_for_name(name: str, count: int = 10) -> list[str]:
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
    print("Hướng dẫn: đặt khuôn mặt vào khung. Khi sẵn sàng, nhấn 'c' để chụp, 'q' để dừng.")
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
            print(f"Captured {idx}/{count}: {filename}")
        elif key == ord('q'):
            print("User canceled capture.")
            break
    cap.release()
    cv2.destroyAllWindows()
    return captured_paths


def add_face_via_webcam_flow() -> None:
    """Flow để thêm khuôn mặt mới: đăng nhập root -> nhập tên -> capture 10 ảnh -> train -> validate on captured images."""
    # 1) kiểm tra root
    if not Path("auth/root.hash").exists():
        print("Chưa có root account. Tạo root trước.")
        create_root_account()
    if not verify_root_login():
        print("Đăng nhập thất bại. Hủy.")
        return
    # 2) nhập tên
    name = input("Nhập tên người cần thêm (no spaces recommended): ").strip()
    if not name:
        print("Tên không hợp lệ.")
        return
    # 3) capture
    captured = capture_images_for_name(name, count=10)
    if len(captured) < 10:
        print("Số ảnh chụp không đủ. Xóa ảnh tạm (nếu có) và thử lại.")
        return
    # 4) train
    print("Bắt đầu huấn luyện (encode)...")
    encode_known_faces(model=args.model)
    # 5) validate the newly captured images
    print("Kiểm tra lại 10 ảnh vừa chụp...")
    # load encodings
    with DEFAULT_ENCODINGS_PATH.open("rb") as f:
        loaded = pickle.load(f)
    bad_count = 0
    for img_path in captured:
        img = face_recognition.load_image_file(img_path)
        locs = face_recognition.face_locations(img, model=args.model)
        encs = face_recognition.face_encodings(img, locs)
        recognized_names = [ _recognize_face(e, loaded) for e in encs ]
        # check that at least one face exists and all recognized as 'name'
        if not recognized_names or any(r != name for r in recognized_names):
            bad_count += 1
            print(f"Validation fail for {img_path}: recognized={recognized_names}")
        else:
            print(f"Validation ok for {img_path}: {recognized_names}")
    if bad_count == 0:
        print(f"✅ Thêm khuôn mặt '{name}' thành công! Tất cả ảnh đều nhận dạng là {name}.")
    else:
        print(f"⚠️ Có {bad_count} ảnh không pass validation. Vui lòng xem lại hoặc chụp lại.")

# Nếu người dùng muốn tạo root nhanh từ CLI
if __name__ == "__main__":
    import sys
    if "--create-root" in sys.argv:
        create_root_account()
    elif "--add-face" in sys.argv:
        add_face_via_webcam_flow()
    # chương trình chính vẫn hoạt động như trước (giữ compat)
