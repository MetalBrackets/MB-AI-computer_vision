from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report



class Metrics_utils:
    """Classe utilitaire recensant les différentes méthodes liées aux scores
    """
    def f1_scores(self, listExpectedEncoded, listPredictedEncoded):
        """Affiche le score F1

        Args:
            listExpectedEncoded (list<int>): Liste des labels réels encodé (normal = 0 , virus = 1, bactérie =2)
            listPredictedEncoded (list<int>): Liste des labels prédits par le model, encodés
        """
        # Calculate F1 score for each class separately
        f1_per_class = f1_score(listExpectedEncoded,
                                listPredictedEncoded, average=None)

        # Calculate micro-average F1 score
        f1_micro = f1_score(listExpectedEncoded,
                            listPredictedEncoded, average='micro')

        # Calculate macro-average F1 score
        f1_macro = f1_score(listExpectedEncoded,
                            listPredictedEncoded, average='macro')

        # Calculate weighted-average F1 score
        f1_weighted = f1_score(listExpectedEncoded,
                               listPredictedEncoded, average='weighted')

        print("F1 score per class:", f1_per_class)
        print("Micro-average F1 score:", f1_micro)
        print("Macro-average F1 score:", f1_macro)
        print("Weighted-average F1 score:", f1_weighted)

    def precision_and_recall(self, listExpectedEncoded, listPredictedEncoded):
        """Affiche la précision et le recall

        Args:
            listExpectedEncoded (list<int>): Liste des labels réels encodé (normal = 0 , virus = 1, bactérie =2)
            listPredictedEncoded (list<int>): Liste des labels prédits par le model, encodés
        """        # Calculate Precision and Recall
        precision = precision_score(
            listExpectedEncoded, listPredictedEncoded, average='weighted')
        recall = recall_score(listExpectedEncoded,
                              listPredictedEncoded, average='weighted')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')

    def print_confusion_matrix(conf_matrix):

        el_xaxis = range(conf_matrix.shape[0])
        el_yaxis = range(conf_matrix.shape[1])
        
        if(len(el_xaxis) == 3):
            axis_labels = ['Normal', 'Virus', 'Bacteria']
        elif(len(el_xaxis) == 2):
            axis_labels = ['Normal', 'Pneumonia']
        else:
            axis_labels = el_yaxis
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in el_xaxis:
            for j in el_yaxis:
                ax.text(
                    x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
        ax.set_xticks(el_xaxis)
        ax.set_yticks(el_yaxis)
        ax.xaxis.set_ticklabels(axis_labels); ax.yaxis.set_ticklabels(axis_labels);
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()

    def print_accuracy_and_loss(self, history):
        """Méthode affichant les courbes de training_accuracy VS validation_accuracy et training_loss VS validation_loss

        Args:
            history (_type_): résultat du model.fit
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(history.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(range(0, len(acc)), acc, label='Training Accuracy')
        plt.plot(range(0, len(val_acc)), val_acc,
        label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(range(0, len(loss)), loss, label='Training Loss')
        plt.plot(range(0, len(val_loss)),
                        val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        
    def cohen_kappa(self,y_true, y_pred): # to what extent 2 models can make the same prediction
        # eg A kappa value of 0.70 indicates substantial agreement between the predictions of Model A and Model B. This means that, beyond what would be expected by chance, the two models tend to make similar predictions for the same patients.
        print("""
        Expose the level of aggrement between 2 raters (it can be models)
        0.81 to 1.00: Almost perfect agreement.
        0.61 to 0.80: Substantial agreement.
        0.41 to 0.60: Moderate agreement.
        0.21 to 0.40: Fair agreement.
        0.00 to 0.20: Slight agreement.
        Less than 0.00: Poor agreement.
              """)
        return print("cohen kappa value is ", cohen_kappa_score(y_true, y_pred))
        
    def precision_recall(self,y_true, y_prob):# impruve readability. Maybe the curve is not well  draw
      precision, recall, _ = precision_recall_curve(y_true, y_prob)
      plt.figure()
      plt.plot(recall, precision, marker='.')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Precision-Recall Curve')
      plt.show()
      
      
    def print_classification_report(self, listExpectedEncoded, listPredictedEncoded):
        class_report = classification_report(listExpectedEncoded, listPredictedEncoded, zero_division=0)
        print(class_report)