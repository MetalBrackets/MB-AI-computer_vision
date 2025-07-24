# Notes à propos du modèle EfficientNet

Sur le `base_model_truncated = Model(inputs=base_model.input, outputs=base_model.layers[22].output)`  
On voit que la `val_accuracy` reste bloqué à 0.33

```
Démarrage de l'entraînement...
Epoch 1/3
218/218 ━━━━━━━━━━━━━━━━━━━━ 136s 618ms/step - accuracy: 0.5022 - loss: 1.0140 - val_accuracy: 0.3333 - val_loss: 1.1441
Epoch 2/3
218/218 ━━━━━━━━━━━━━━━━━━━━ 135s 608ms/step - accuracy: 0.5728 - loss: 0.9382 - val_accuracy: 0.3333 - val_loss: 1.3821
Epoch 3/3
218/218 ━━━━━━━━━━━━━━━━━━━━ 136s 615ms/step - accuracy: 0.5802 - loss: 0.8736 - val_accuracy: 0.3333 - val_loss: 1.7820
Historique accuracy de train: [0.512269914150238, 0.5870398879051208, 0.6096625924110413]
Historique accuracy de validation: [0.3333333432674408, 0.3333333432674408, 0.3333333432674408]
Historique de perte de train: [0.9950289130210876, 0.9090233445167542, 0.8449075222015381]
Historique de perte de validation: [1.1441034078598022, 1.382123351097107, 1.7819710969924927]

Dernière valeur accuracy de train: 0.6096625924110413
Dernière valeur accuracy de validation: 0.3333333432674408
Dernière valeur de perte de train: 0.8449075222015381
Dernière valeur de perte de validation: 1.7819710969924927
```

Les tests effectués pour débuger la val_accuracy :

- utilisation d'un autre algo d'optimisation que Adam -> RMSprop
- learning_rate=0.00001
- flip appliquée aléatoirement sur les données de train
- ajout d'une couche de droput pour la régulation
- entrainer le modèle avec la totalité de ses couches (240)

Pour l'instant rien n'y fait, la val_accuracy reste bloqué à 0.33

**Solution** -> Tester un nouveau modèle plus simple
