from pathlib import Path
import face_recognition


DEFAULT_ENCODINGS_PATH = Path("output/encoding.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_know_faces(model :str = "hog",encoding_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image,model=model)
        face_encodings = face_recognition.face_encodings(image,face_locations)
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding) 
    return names, encodings

names,encodings = encode_know_faces()
print(names)
print(encodings)
