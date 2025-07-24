import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generator(data_dir, batch_size, color_mode, seed, shuffle, classes=None):
    datagen = ImageDataGenerator(rescale=1./255)
    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        seed=seed,
        shuffle=shuffle,
        classes=classes 
    )
    return data_generator

# afficher les formes (shape) des datasets
def print_shapes_and_labels(generator):
    # next(t) récupère le prochain lot de 32 images et labels (défini par batch_size)
    images, labels = next(generator)
    print(f"\nImages : {images.shape}")  # output (batch_size, height, width, channels)
    print(f"Labels : {labels.shape}")  # output (batch_size, num_classes)
    # print("Label one-hot :")
    # optionnel : afficher les labels de train
    # print(labels) # output [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.] ...]


# afficher le radio des classes de poumons dans chaque dataset (train, validation et test)
def get_nb_img_by_classes(base_dir):
    nb_img_by_classe = {}
    # nb images classe normal
    normal_dir = os.path.join(base_dir, "normal")
    nb_normal = len([f for f in os.listdir(normal_dir) if os.path.isfile(os.path.join(normal_dir, f))])
    # nb images classe bacteria
    bacteria_dir = os.path.join(base_dir, "bacteria")
    nb_bacterie = len([f for f in os.listdir(bacteria_dir) if os.path.isfile(os.path.join(bacteria_dir, f))])
    # nb images classe virus
    virus_dir = os.path.join(base_dir, "virus")
    nb_virus = len([f for f in os.listdir(virus_dir) if os.path.isfile(os.path.join(virus_dir, f))])

    nb_img_by_classe["nb normal"] = nb_normal  # nb de poumon normal (sain)
    nb_img_by_classe["nb bacteria"] = nb_bacterie  # nb de poumon malade par baterie
    nb_img_by_classe["nb virus"] = nb_virus  # nb de poumon malade par virus

    return nb_img_by_classe
    # output :
    #        Train : {'nb normal': 1341, 'nb bacteria': 2530, 'nb virus': 1345}
    #        Val : {'nb normal': 8, 'nb bacteria': 8, 'nb virus': 8}
    #        Test : {'nb normal': 234, 'nb bacteria': 242, 'nb virus': 140}



# retourner class_name en inversant les clés et les valeurs
def get_class_names(class_indices):
    class_names = {}
    # inverser les clés et les valeurs
    # {'bacteria': 0, 'normal': 1, 'virus': 2} -> {0: 'bacteria', 1: 'normal', 2: 'virus'}
    for class_name, class_index in class_indices.items():
        class_names[class_index] = class_name
    return class_names


# afficher une grille de 4x4
def plot_4_by_4_images(images, labels, class_names):
    fig = plt.figure(figsize=(10, 10))  # taille de la figure
    for i in range(4):
        for j in range(4):
            # sous-graphique, 4lignes, 4 col, index unique
            ax = fig.add_subplot(4, 4, 4 * i + j + 1)  # "+1" -> 1er index en matplotlib
            ax.matshow(images[4 * i + j + 1])
            class_index = labels[4 * i + j].argmax()
            ax.set_title(class_names[class_index])
            ax.axis("off")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()