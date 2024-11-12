from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Lista de URLs de las imágenes
urls = [
    "http://assets.myntassets.com/v1/images/style/properties/Raymond-Men-Raymond-Leather-Belts-White-Belts_f942698ad78d2e13a4007024e6c73f3c_images.jpg",
    "http://assets.myntassets.com/v1/images/style/properties/US-Polo-Assn-Kids-Boys-Orange-T-shirt_a7c1b583f1c04c05f9bd6ffca5358241_images.jpg",
    "http://assets.myntassets.com/v1/images/style/properties/Reebok-Men-REALFLEX-OPTIMAL-White-Sports-Shoes_6e49515f918bc0fb66df906799386112_images.jpg",
    "http://assets.myntassets.com/v1/images/style/properties/Reid---Taylor-Men-Blue-Striped-Shirt_b49d226c8a40fc2049dfb6280149e4ff_images.jpg",
    "http://assets.myntassets.com/v1/images/style/properties/Adidas-Men-SUKOI-White-Sports-Shoes_2d184f92b877bb5f06e2ba7c725590f7_images.jpg"
]

# Descargar y mostrar cada imagen
images = []
for url in urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    images.append(img)

# Crear una cuadrícula de 1x5 para mostrar las imágenes
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for ax, img in zip(axes, images):
    ax.imshow(img)
    ax.axis('off')  # Quitar los ejes para una mejor visualización

plt.show()
