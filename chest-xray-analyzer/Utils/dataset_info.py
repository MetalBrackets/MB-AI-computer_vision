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

# Afficher les formes (shape) des datasets
def print_shapes_and_labels(generator):
    # next(t) récupère le prochain lot de 32 images et labels (défini par batch_size)
    images, labels = next(generator)
    print(f"\nImages : {images.shape}")  # output (batch_size, height, width, channels)
    print(f"Labels : {labels.shape}")  # output (batch_size, num_classes)
    # print("Label one-hot :")
    # Optionnel : afficher les labels de train
    # print(labels) # output [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.] ...]


# Afficher le radio des classes de poumons dans chaque dataset (train, validation et test)
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



# Retourner class_name en inversant les clés et les valeurs
def get_class_names(class_indices):
    class_names = {}
    # inverser les clés et les valeurs
    # {'bacteria': 0, 'normal': 1, 'virus': 2} -> {0: 'bacteria', 1: 'normal', 2: 'virus'}
    for class_name, class_index in class_indices.items():
        class_names[class_index] = class_name
    return class_names


# Afficher une grille de 4x4
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


# Afficher le nombre de couches dans le modèle
def print_layer_info(model):
    print(f"nb total de couches dans le modèle : {len(model.layers)}")
    print(f"Détail :")
    for i, layer in enumerate(model.layers):
        layer_name = layer.__class__.__name__
        if hasattr(layer, 'units'):
            print(f"Layer {i+1} ({layer_name}) : {layer.units} neurones")
        elif hasattr(layer, 'filters'):
            print(f"Layer {i+1} ({layer_name}) : {layer.filters} filtres")
        else:
            print(f"Layer {i+1} ({layer_name})")

# Calculer le temps d'entrainement du modèle
def print_training_duration(total_start_time, total_end_time):
    total_training_duration = total_end_time - total_start_time
    minutes = int(total_training_duration // 60)
    seconds = int(total_training_duration % 60)
    print(f"\n⏱️ Temps total d'entraînement : {minutes} minutes et {seconds} secondes")
    return total_training_duration


#  Histogramme Empilé
def plot_stacked_bar_chart(base_dirs, titles):
    labels = ['Normal', 'Bacteria', 'Virus']
    data = {label: [] for label in labels}

    # Collect data for each label from each dataset
    for directory in base_dirs.values():
        nb_img_by_class = get_nb_img_by_classes(directory)
        for label in labels:
            data[label].append(nb_img_by_class['nb ' + label.lower()])

    fig, ax = plt.subplots(figsize=(10, 7))
    bottoms = np.zeros(len(titles))
    
    # Create the bar chart
    for label, color in zip(labels, ['skyblue', 'salmon', 'limegreen']):
        ax.bar(titles, data[label], bottom=bottoms, label=label, color=color)
        bottoms += np.array(data[label])
    
    ax.set_ylabel('Nombre d\'images')
    ax.set_title('Distribution des classes par dataset')
    ax.legend()

    plt.show()
            
def show_population_distribution(labels_list,labels_size ):
    labels = labels_list
    sizes = labels_size

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    

def display_img(img):
    plt.imshow(img,cmap='gray')
    plt.show()
