import os

base_path = "dataset/chest_xray"

folders = ["train", "test", "val"]

for folder in folders:
    path = os.path.join(base_path, folder)

    print("\n==============================")
    print(f"{folder.upper()} DATASET")
    print("==============================")

    if not os.path.exists(path):
        print(f"{folder} folder not found!")
        continue

    for disease_class in ["NORMAL", "PNEUMONIA"]:
        class_path = os.path.join(path, disease_class)

        if os.path.exists(class_path):
            files = os.listdir(class_path)
            print(f"{disease_class}: {len(files)} images")
        else:
            print(f"{disease_class} folder not found!")