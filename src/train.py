# Train model

from ultralytics import YOLO
model = YOLO('./yolov8n.pt')

results = model.train(data='../datasets/mydataset.yml',
                      epochs=200,
                      project='../output')

# --- Augmentering ---

#flipud=0.2, # Speglar bilden vertikalt med 20% sannolikhet – ökar variation
#mixup=0.2, # Blandar två bilder och deras etiketter – hjälper generalisering
#copy_paste=0.1, # Klistrar in objekt från andra bilder – skapar fler träningsexempel
#degrees=15.0, # Roterar bilden slumpmässigt upp till ±15 grader
#shear=5.0, # Förvränger bilden lätt – simulerar olika kameravinklar

# --- Regularisering ---

#dropout=0.2, # Stänger av 20% av neuronen slumpmässigt under träning – motverkar overfitting
#label_smoothing=0.1 # Mjukar upp etiketterna (t.ex. 1.0 → 0.9) – minskar överkonfidens)
