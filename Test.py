import os
count = 0

for f in os.listdir("labels/train"):
    with open(f"labels/train/{f}") as file:
        if file.read().strip() != "":
            count += 1

print("Pliki z obiektami:", count)

imgs = os.listdir("images_sliced/train")
labels = os.listdir("labels/train")

print(len(imgs), len(labels))