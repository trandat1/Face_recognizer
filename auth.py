from pathlib import Path
import argparse
import os
import pickle
from collections import Counter
from datetime import datetime, timedelta
from core import Core
import cv2
import face_recognition


class Auth():
    def _hash_password(self,password: str, salt: str) -> str:
        import hashlib
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


    def create_root_account(self) -> None:
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
        hashed = self._hash_password(pwd1, salt)
        # lưu dưới dạng salt$hash
        root_file.write_text(salt + "$" + hashed)
        print("✅ Tạo root account thành công.")


    def verify_root_login(self) -> bool:
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
        return self._hash_password(pwd, salt) == saved_hash


