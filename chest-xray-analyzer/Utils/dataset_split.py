class dataset_split:
  """Classe utilitaire permettant de réorganiser le dataset
  """
  def __init__(self, folder_image_train, folder_image_test, folder_image_val):
    """Méthode d'initialisation

    Args:
        folder_image_train (string): path du dossier train
        folder_image_test (_type_): path du dossier test
        folder_image_val (_type_): path du dossier val
    """
    self.folder_image_train = folder_image_train
    self.folder_image_test = folder_image_test
    self.folder_image_val = folder_image_val
    self.list_folders = [folder_image_train, folder_image_test, folder_image_val]
    
  def images_resizer(self, width, height, color, folder_img, folder_image_final):
    """Permet de copier le dataset avec la width et height donné

    Args:
        width (int): largeur de l'image
        height (int): hauteur de l'image
        color (int): 1 pour du noir et blanc, 3 pour du rgb
        folder_img (string): path du dossier à resize (qui contient train, test et val)
        folder_image_final (string): path du dossier contenant les images retouchées
    """
    for dossier in os.listdir(folder_img):
      typeJeuDeDonnee = folder_img.split('/')[-2]
      newpath = folder_image_final + typeJeuDeDonnee + "/" + dossier 
      if not os.path.exists(newpath):
        os.makedirs(newpath)
        for filename in os.listdir(folder_img + "/" + dossier):
          if (filename.split('.')[-1] == 'jpeg'):
            img = tf.keras.preprocessing.image.load_img(folder_img + "/" + dossier + "/" + filename)
            data = tf.keras.preprocessing.image.img_to_array(img)
            img = np.array(data).astype('float32')
            img = resize(img,(width , height, color))
            img = tf.keras.preprocessing.image.array_to_img(img)
            picture = img.save(newpath + "/" + filename)
            
  def virus_bactera_folder_creator(self, dossier):
    """Créer les dossiers virus et bactérie

    Args:
        dossier (string): path du dossier où créer les sous dossiers virus/ bacteria (dossier test, train ou val)
    """
    for filename in os.listdir(dossier):
        virusPath = dossier + "/" + "virus"
        bacteriaPath = dossier + "/" + "bacteria"
        listPaths = [virusPath, bacteriaPath]
        for path in listPaths:
          if not os.path.exists(path):
              os.makedirs(path)
    
  def classify(self, image_path):
    """Positionne une image dans le dossier virus ou bactérie en fonction de son nom

    Args:
        image_path (string): path de l'image à trier
    """
    pathSplit = image_path.split('/')
    image_name = pathSplit[-1]
    nomSplit = image_name.split('_')
    for element in nomSplit:
      if(element.lower() == "bacteria"):
        newPath = '/'.join(pathSplit[:-1]) + "/bacteria" + "/" + image_name
        os.rename(image, newPath)
      elif(element.lower() == "virus"):
        newPath = '/'.join(pathSplit[:-1]) + "/virus" + "/" + image_name
        os.rename(image, newPath)
    
  def images_split_bacteria_virus(self):
    """Fonction principale à utiliser après instanciation de la classe
    
    Enchaine la fonction de création de dossier virus/bacteria sur les 3 emplacements
    des dossiers train, test et val. Trie ensuite chacune des photos du dossier.
    """
    for folder in self.list_folders:
      self.virus_bactera_folder_creator(folder)
      for img in os.listdir(folder):
        if(img.lower() != "virus" and img.lower() != "bacteria"):
          pathImage = dossier + "/" + img
          classify(pathImage) 