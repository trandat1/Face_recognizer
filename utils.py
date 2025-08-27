from pathlib import Path
import os
import cv2
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


class SecurityUtils:
    # Load config từ .env hoặc biến môi trường
    ALERT_SENDER = os.getenv("ALERT_SENDER", "your_email@gmail.com")
    ALERT_PASSWORD = os.getenv("ALERT_PASSWORD", "")  # App password nếu Gmail
    ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", ALERT_SENDER)
    ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "300"))  # 5 phút mặc định
    LAST_EMAIL_MARKER = Path("logs/last_email.txt")

    @classmethod
    def _can_send_email_now(cls) -> bool:
        """Kiểm tra có thể gửi email hay chưa (tôn trọng cooldown)."""
        if not cls.ALERT_PASSWORD or not cls.ALERT_SENDER:
            return False
        if not cls.LAST_EMAIL_MARKER.exists():
            return True
        try:
            ts = cls.LAST_EMAIL_MARKER.read_text().strip()
            last_time = datetime.fromisoformat(ts)
        except Exception:
            return True
        return datetime.now() - last_time >= timedelta(seconds=cls.ALERT_COOLDOWN_SECS)

    @classmethod
    def _touch_email_marker(cls) -> None:
        """Đánh dấu thời gian gửi email gần nhất."""
        cls.LAST_EMAIL_MARKER.parent.mkdir(exist_ok=True)
        cls.LAST_EMAIL_MARKER.write_text(datetime.now().isoformat())

    @classmethod
    def send_alert_email(cls, image_path: str, receiver: str | None = None) -> None:
        """Gửi email cảnh báo kèm ảnh Unknown."""
        if not cls._can_send_email_now():
            print("[Email] Đang trong thời gian cooldown, bỏ qua gửi email.")
            return

        sender = cls.ALERT_SENDER
        password = cls.ALERT_PASSWORD
        to_addr = receiver or cls.ALERT_RECEIVER

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

            cls._touch_email_marker()
            print(f"📧 Email cảnh báo đã gửi kèm ảnh: {image_path}")

        except Exception as e:
            print(f"[Email] Lỗi gửi email: {e}")

    @staticmethod
    def log_intrusion(name: str = "Unknown"):
        """Ghi log xâm nhập vào file intrusion.log."""
        logline = f"[{datetime.now().isoformat()}] Detected: {name}\n"
        Path("logs").mkdir(exist_ok=True)
        with open("logs/intrusion.log", "a", encoding="utf-8") as f:
            f.write(logline)
        print(logline.strip())

    @staticmethod
    def save_intruder_image(frame, bounding_box) -> str:
        """Lưu ảnh của intruder (cắt từ bounding box)."""
        top, right, bottom, left = bounding_box
        margin = 20
        h, w = frame.shape[:2]
        t = max(0, top - margin)
        l = max(0, left - margin)
        b = min(h, bottom + margin)
        r = min(w, right + margin)
        face_image = frame[t:b, l:r]

        Path("intruder").mkdir(exist_ok=True)
        filename = Path("intruder") / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(filename), face_image)
        print(f"[Saved] Ảnh Unknown: {filename}")
        return str(filename)

    @classmethod
    def alert_intruder(cls, frame, bounding_box, name: str = "Unknown"):
        """Quy trình cảnh báo intruder: lưu ảnh, ghi log, gửi email."""
        img_path = cls.save_intruder_image(frame, bounding_box)
        cls.log_intrusion(name)
        cls.send_alert_email(img_path)
        return img_path
