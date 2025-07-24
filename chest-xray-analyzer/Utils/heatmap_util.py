import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report


class Heatmap_utils:
    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        """ Méthode pour générer une heatmap des poids les plus important d'un model

        Args:
            img_array (_type_): _description_
            model (_type_): _description_
            last_conv_layer_name (_type_): _description_
            pred_index (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(
                last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        plt.hist(grads.numpy().flatten())

        print("Last conv layer output shape:", last_conv_layer_output.shape)
        print("Gradients shape:", grads.shape)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print(pooled_grads.shape)

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        print(last_conv_layer_output.shape)
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    

    def make_gradcam_heatmap2(self, img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap[0]


    def overlay_heatmap(img, heatmap, alpha=0.5, colormap='viridis'):
        """Génère une image avec une heatmap représentant l'emplacement des pixels ayant les poids les plus important du modèle.

        Args:
            img (_type_): image originale encodée
            heatmap (_type_): heatmat retournée par make_gradcam_heatmap
            alpha (float, optional): _description_. Defaults to 0.5.
            colormap (str, optional): _description_. Defaults to 'viridis'.

        Returns:
            _type_: _description_
        """
        if len(img.shape) == 4:
            img = img[0]
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        # img = np.uint8(255 * img)

        # Use a colormap to colorize the heatmap
        colormap = plt.get_cmap(colormap)
        heatmap = colormap(heatmap)
        # print(img.shape)
        heatmap = np.delete(heatmap, 3, 2)  # Delete the alpha channel

        # Resize heatmap to match the size of the original image
        heatmap = tf.image.resize(heatmap, (img.shape[0], img.shape[1]))
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(heatmap)
        # axarr[0].axis('off')
        # axarr[1].imshow(tf.keras.utils.array_to_img(img))
        # axarr[1].axis('off')
        # plt.show()
        heatmap = tf.keras.utils.img_to_array(heatmap)
        # print(heatmap.shape)

        heatmap = cv2.applyColorMap(np.uint8(-255*heatmap), cv2.COLORMAP_JET)
        # Superimpose the heatmap on the original image
        superimposed_img = heatmap * alpha + img
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

        return superimposed_img

    def import_original_image(path, size):
        """Retourne un tenseur de l'image

        Args:
            path (String): emplacement de l'image
            size ((int, int)): (width, height) en pixel

        Returns:
           tensor
        """
        return img_to_array(load_img(path, target_size=size))

    def import_grayscale_image(path, size):
        """Retourne un tenseur de l'image en noir et blanc

        Args:
            path (String): emplacement de l'image
            size ((int, int)): (width, height) en pixel

        Returns:
            tensor
        """
        return img_to_array(load_img(path,  color_mode="grayscale", target_size=size))

    def heatmaps(dossier_images, img_height, img_width, model):
        """Créer 10 heatmaps sur 10 images aléatoire du dossier d'images

        Args:
            dossier_images (string): path du dossier contenant les images
            img_height (integer): height pris par le modèle
            img_width (integer): width pris par le modèle
            model (tensorflow model): modèle entrainé

        Returns:
            _type_: _description_
        """
        heatmaps = []
        # Remove last layer's softmax
        model.layers[-1].activation = None

        for num in range(1, 11):

            image = os.listdir(DOSSIER_TEST_BAC)[randint(
                0, len(os.listdir(DOSSIER_TEST_BAC)))]
            image_path = DOSSIER_TEST_BAC + image
            image_size = (img_height, img_width)

            img_origin = Image_utils.import_original_image(
                image_path, image_size)
            img_gray = Image_utils.import_grayscale_image(
                image_path, image_size)

            img = output = tf.expand_dims(img_gray, 0)
            # Print what the top predicted class is
            preds = model.predict(img)
            # print("Predicted:", preds[0])

            # Generate class activation heatmap
            heatmap = Image_utils.make_gradcam_heatmap(
                img, model, "last_conv_layer_name")
            heatmaps.append([img_origin, heatmap])

        return heatmaps
