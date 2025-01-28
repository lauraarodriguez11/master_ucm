//Actualizamos todos los documentos añadiendo el campo 'image_url'
//para tener toda la información recogida en un mismo .json

const { MongoClient } = require('mongodb');

async function updateProductsWithImages() {
  const uri = "mongodb://localhost:27017";
  const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

  try {
    // Conectar a MongoDB
    await client.connect();

    const db = client.db('fashion_products'); 
    const productosCollection = db.collection('styles');
    const imagenesCollection = db.collection('images'); 

    // Obtener todas las imágenes de la colección
    const imagenes = await imagenesCollection.find({}).toArray();

    // Iterar sobre cada imagen y actualizar el producto correspondiente
    for (const imagen of imagenes) {
      const productId = parseInt(imagen.filename.split('.')[0]); // Obtener el id del producto del nombre del archivo

      // Actualizar el producto correspondiente en la colección 'productos'
      await productosCollection.updateOne(
        { id: productId }, // Buscar el producto por su id
        { $set: { image_url: imagen.link } } // Establecer el campo image_url con el enlace de la imagen
      );

    }

    console.log("Actualización de productos con imágenes completada.");
  } catch (error) {
    console.error("Error al actualizar productos:", error);
  } finally {
    // Cerrar la conexión a MongoDB
    await client.close();
  }
}

updateProductsWithImages();