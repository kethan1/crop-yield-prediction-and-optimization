import os
import cv2
import numpy as np
from sklearn import neighbors
import sklearn.model_selection

datafolder = "data/PlantVillage/Potato"
plant_paths = [os.path.join(datafolder, "early_blight"), os.path.join(datafolder, "healthy"), os.path.join(datafolder, "late_blight")]

images = {}

for plant_path in plant_paths:
    path = os.path.basename(os.path.normpath(plant_path))
    print(path)
    images[path] = []
    for image in os.listdir(plant_path):
        img = cv2.imread(os.path.join(datafolder, path, image))
        if img is None:
            print(os.path.join(datafolder, path, image))
        images[path].append(img)

labels = np.asarray([each_character for character, image_lst in images.items() for each_character in len(image_lst) * [character]])
# labels[labels == "bacterial_spot"] = 0
labels[labels == "early_blight"] = 0
labels[labels == "healthy"] = 1
labels[labels == "late_blight"] = 2
# labels[labels == "leaf_mold"] = 4
# labels[labels == "mosaic_virus"] = 5
# labels[labels == "septoria_leaf_spot"] = 6
# labels[labels == "spider_mites"] = 7
# labels[labels == "target_spot"] = 8
# labels[labels == "yellow_leaf_curl_virus"] = 9
data = np.asarray([image.flatten() for image_lst in images.values() for image in image_lst])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.2, shuffle=True)

clf = neighbors.KNeighborsClassifier()
# Train the classifier
clf.fit(x_train, y_train)

# Get the accuracy of the classifier basing on the test data
accuracy = clf.score(x_test, y_test)
print(accuracy)
