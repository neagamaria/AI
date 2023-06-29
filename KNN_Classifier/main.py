import csv
import numpy as np
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

train_img_labels = {}
train_labels = []
test_img = []
val_img_labels = {}
ordered_val_images = []
val_labels = []
train_images = []
test_images = []
val_images = []

# citire nume imagini si etichete pentru imaginile de antrenare
f1 = open('data/train.csv', 'r')
input_train_img = csv.reader(f1)
next(input_train_img)
for line in input_train_img:
    img, label = line
    train_img_labels[img] = int(label)

# citire nume imagini si etichete pentru imaginile de validare
f3 = open('data/val.csv', 'r')
input_val_set = csv.reader(f3)
next(input_val_set)
for line in input_val_set:
    img, label = line
    val_img_labels[img] = int(label)

# citire nume imagini pentru imaginile de test
f2 = open('data/test.csv', 'r')
input_test_img = csv.reader(f2)
next(f2)  # ignore the header
for line in input_test_img:
    test_img.append(line[0])


# citire efectiva a imaginilor
trainig_folder_path = "data/train_images"
for image_name in os.listdir(trainig_folder_path):
    current_path = os.path.join(trainig_folder_path, image_name)
    current_image = Image.open(current_path)
    train_labels.append(train_img_labels[image_name]) # pentru a obtine label-urile in ordinea imaginilor
    train_images.append(np.array(current_image).flatten()) # condensare date
    current_image.close()
train_images = np.array(train_images)

test_folder_path = "data/test_images"
for image_name in os.listdir(test_folder_path):
    current_path = os.path.join(test_folder_path, image_name) # calea imaginea curenta
    current_image = Image.open(current_path)
    test_images.append(np.array(current_image).flatten())
    current_image.close()
test_images = np.array(test_images)

val_folder_path = "data/val_images"
for image_name in os.listdir(val_folder_path):
    current_path = os.path.join(val_folder_path, image_name)
    current_image = Image.open(current_path)
    val_labels.append(val_img_labels[image_name])
    ordered_val_images.append(image_name)
    val_images.append(np.array(current_image).flatten())
    current_image.close()
val_images = np.array(val_images)

# normalizarea
scaler = MinMaxScaler()
train_images = scaler.fit_transform(train_images)
test_images = scaler.transform(test_images)
val_images = scaler.transform(val_images)

# definirea modelului KNN
knn = KNeighborsClassifier(n_neighbors=7, metric="hamming")
knn.fit(train_images, train_labels)

val_predictions = knn.predict(val_images)
val_accuracy = np.mean(val_predictions == val_labels)
val_precision = precision_score(val_labels, val_predictions, average='weighted', zero_division=1.0) # precizia obtinuta pentru predictii
print(f'Accuracy for validation data: {val_accuracy}\nPrecision: {val_precision}')

# matricea de confuzie
conf_mx = confusion_matrix(val_labels, val_predictions)
print(f'Matricea de confuzie\n:{conf_mx}')

# alegere cateva dintre etichetele de validare pentru a afisa o submatrice
chosen_labels = np.random.choice(knn.classes_, size=min(10, len(knn.classes_)), replace=False)

sub_conf_mx = conf_mx[np.isin(knn.classes_, chosen_labels)][:, np.isin(knn.classes_, chosen_labels)] # feliere matrice
show = ConfusionMatrixDisplay(sub_conf_mx)
show.plot()
plt.show()

# predictiile pentru datele de testare
final_predictions = knn.predict(test_images)

# afisare in fisier csv
fw = open('data/sample_submission.csv', 'w', newline='')
write_in_csv = csv.writer(fw)
write_in_csv.writerow(["Image", "Class"])
for i in range(len(final_predictions)):
    write_in_csv.writerow([test_img[i], final_predictions[i]])