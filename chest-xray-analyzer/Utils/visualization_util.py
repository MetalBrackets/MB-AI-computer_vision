
import matplotlib.pyplot as plt

# afficher le nombre de couches dans le modèle
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

def plot_accuracy_and_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # sous-graphique pour l'accuracy
    plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, 1er graphique
    plt.plot(epochs_range, acc, 'o-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'o-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # sous-graphique pour la loss
    plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, 2ème graphique
    plt.plot(epochs_range, loss, 'o-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'o-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout() 
    plt.show()


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
