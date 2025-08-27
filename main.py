import argparse
from core import Core
from face_add import FaceAdd
from auth import Auth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--train", action="store_true", help="Train on input data")
    parser.add_argument("--validate", action="store_true", help="Validate trained model")
    parser.add_argument("--test", action="store_true", help="Test with an image")
    parser.add_argument("-f", "--file", type=str, help="Path to test image")
    parser.add_argument("--webcam", action="store_true", help="Run recognition with webcam")
    parser.add_argument("--create-root", action="store_true", help="Create root account")
    parser.add_argument("--add-face", action="store_true", help="Add new face via webcam")
    parser.add_argument("--model", type=str, default="hog", help="Face detection model: hog hoặc cnn")

    args = parser.parse_args()

    # Khởi tạo core
    core = Core()

    if args.create_root:
        Auth.create_root_account()

    elif args.add_face:
        # bắt buộc đăng nhập root trước
        if Auth.verify_root_login():
            face_add = FaceAdd()
            face_add.add_face_via_webcam_flow(model=args.model)
        else:
            print("❌ Root login failed!")

    elif args.train:
        core.encode_known_faces(model=args.model)

    elif args.validate:
        core.validate(model=args.model)

    elif args.test:
        if not args.file:
            raise SystemExit("--test cần kèm đường dẫn ảnh bằng -f")
        core.recognize_faces_from_image(image_location=args.file, model=args.model)

    elif args.webcam:
        core.recognize_from_webcam(model=args.model)
