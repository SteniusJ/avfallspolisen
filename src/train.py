# Train model

from ultralytics import YOLO
model = YOLO('./runs/output/train4/weights/best.pt')

results = model.train(data='../datasets/mydataset.yml',
                      epochs=100,
                      flipud=0.5,
                      mixup=0.4,
                      copy_paste=0.5,
                      degrees=70.0,
                      shear=30.0,
                      hsv_v=0.8,
                      hsv_h=0.3,
                      translate=0.3,  
                      dropout=0.2,
                      cutmix=0.3,
                      mosaic=0.8,
                      pretrained=True,
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
