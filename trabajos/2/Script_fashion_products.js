// Borrado
//db.styles.drop();
db.styles.find();


// Ejercicios sobre Inserción

//1. Insertar un producto
db.styles.insertOne({
    "id": 60000,
    "gender": "Women",
    "masterCategory": "Accessories",
    "subCategory": "Hats",
    "articleType": "Hats",
    "baseColour": "Black",
    "season": "Spring",
    "year": 2024,
    "usage": "Formal",
    "productDisplayName": "Beret",
    "image_url": "http://assets.myntassets.com/v1/images/style/properties/60000_images.jpg"
});

//2. Borrar el producto que acabamos de añadir
db.styles.deleteOne({ 
    "id": 60000,
    "gender": "Women",
    "masterCategory": "Accessories",
    "subCategory": "Hats",
    "articleType": "Hats",
    "baseColour": "Black",
    "season": "Spring",
    "year": 2024,
    "usage": "Formal",
    "productDisplayName": "Beret",
    "image_url": "http://assets.myntassets.com/v1/images/style/properties/60000_images.jpg"
});



// Ejercicios sobre Actualización

//1. Actualizar la categoría de los relojes para hombre a 'Luxury Accesories'
db.styles.updateMany(
    { "subCategory": "Watches", "gender": "Men" }, 
    { $set: { "masterCategory": "Luxury Accessories" } } 
);

//2. Volver a dejar la categoría como estaba
db.styles.updateMany(
    { "subCategory": "Watches", "gender": "Men" }, 
    { $set: { "masterCategory": "Accessories" } } 
);

//3. Contar productos que no tengan url para comprobar que los hemos actualizado todos correctamente,
//   y de no existir imagen para algún producto escribir 'Undefined' (aunque sabemos que todos lo tienen)
db.styles.find({ "image_url": { $exists: false } }).count();
db.styles.updateMany(
  { $or: [ { "image_url": "" }, { "image_url": null } ] }, 
  { $set: { "image_url": "Undefined" } } 
);



// Ejercicios sobre Proyección y Filtrado

//1. Mostrar sólo los productos con categoría 'Apparel' 
//   enseñando únicamente los campos productDisplayName e image_url
db.styles.find(
    { "masterCategory": "Apparel" }, 
    { "productDisplayName": 1, "image_url": 1, "_id": 0 } 
);

//2. Ordenar por año de forma descendente y limitamos la salida a las 50 primeras filas
db.styles.find({}).sort({ year: -1}).limit(50); 

//3. Ordenamos por año de forma descendente y limitamos la salida a las 50 segundas filas
db.styles.find({}).sort({ year: -1}).skip(50).limit(50);

//4. Descartar productos que sean de años mayores de 2015 y menores de 2014 ($nor)
db.styles.find({ $nor: [ { "year" : { $lt : 2014 } }, { "year" : { $gt : 2015 } } ] });


// Ejercicios sobre Pipeline de Agregación

//1. Mostrar los 10 tipos de producto que tienen más variedad 
db.styles.aggregate([
  { $group: { _id: "$articleType", totalProductos: { $sum: 1 } } },
  { $sort: { totalProductos: -1 } },
  { $limit: 10 }
]);

//2. Mostrar las 15 combinaciones baseColour-articleType predominantes y las menos comunes
db.styles.aggregate([
  { $group: { _id: { subCategory: "$articleType", baseColour: "$baseColour" }, totalProductos: { $sum: 1 } } },
  { $sort: { totalProductos: -1 } },
  { $limit: 15 }
]);

db.styles.aggregate([
  { $group: { _id: { subCategory: "$articleType", baseColour: "$baseColour" }, totalProductos: { $sum: 1 } } },
  { $sort: { totalProductos: 1 } },
  { $limit: 15 }
]);

//3. Mostrar el año u años con más y con menos productos
var fase1 = { $group: { "_id": "$year", "cuenta": { $sum: 1 } } };
var fase2 = { $group: { "_id": "$cuenta", "anios": { $push: "$_id" } } };
var sortMax = { $sort: { "_id": -1 } }; // Para encontrar el máximo número de productos
var sortMin = { $sort: { "_id": 1 } }; // Para encontrar el mínimo número de productos
var limit = { $limit: 1 };
var unwind = { $unwind: "$anios" };

var pipelineMax = [ fase1, fase2, sortMax, limit, unwind ];
var pipelineMin = [ fase1, fase2, sortMin, limit, unwind ];

db.styles.aggregate([
  {
    $facet: {
      "maxYear": pipelineMax,
      "minYear": pipelineMin
    }
  }
]);

// Nos sale como año 'null' porque hay un producto que no le viene asignado ningún año, 
// entonces vamos a añadir un match al principio para evitar esto
var matchValidYear = { $match: { year: { $ne: null } } };
var fase1 = { $group: { "_id": "$year", "cuenta": { $sum: 1 } } };
var fase2 = { $group: { "_id": "$cuenta", "anios": { $push: "$_id" } } };
var sortMax = { $sort: { "_id": -1 } }; 
var sortMin = { $sort: { "_id": 1 } }; 
var limit = { $limit: 1 };
var unwind = { $unwind: "$anios" };

var pipelineMax = [ matchValidYear, fase1, fase2, sortMax, limit, unwind ];
var pipelineMin = [ matchValidYear, fase1, fase2, sortMin, limit, unwind ];

db.styles.aggregate([
  {
    $facet: {
      "maxYear": pipelineMax,
      "minYear": pipelineMin
    }
  }
]);
    
//4. Agrupar por temporada y contar cuántos productos se lanzaron en cada una, ordenando de forma descendente
db.styles.aggregate( [ 
    { $group: { "_id": "$season", "cuenta": { $sum: 1 } } }, 
    { $sort: { "cuenta" : -1 } } ] ); 

//5. Mostrar los 5 tipos de artículo de categoría Footwear de los que hay más productos
var fasefilter={ $match: { "masterCategory": "Footwear" } }
var fasegroup={ $group: { "_id": "$articleType", "cuenta": { $sum: 1 } } }
var fasesort={ $sort: { "cuenta": -1 } }
var faselimit={ $limit: 5 }
var etapas=[ fasefilter, fasegroup, fasesort, faselimit ]
db.styles.aggregate(etapas);

//6. Mostramos las subcategorías junto con el número de productos que tiene cada subcategoría 
//   y la información del producto más reciente en esa subcategoría
db.styles.aggregate([
    { $sort: { "year": -1 } },
    { $group: { 
        _id: "$subCategory", 
        totalProductos: { $sum: 1 }, 
        mostRecentProduct: { $first: "$$ROOT" } 
    }},
    { $sort: { "_id": 1 } }, 
    { $project: {
        _id: 1, 
        totalProductos: 1, 
        "mostRecentProduct.productDisplayName": 1, 
        "mostRecentProduct.image_url": 1,
        "mostRecentProduct.year": 1
    }}
]);

//7. Agrupar los productos por género y color base, mostrando los 5 colores más populares para cada género.
db.styles.aggregate([
  { $group: { 
      _id: { gender: "$gender", baseColour: "$baseColour" },
      totalProductos: { $sum: 1 }
    }
  },
  { $sort: { totalProductos: -1 } },
  { $group: {
      _id: "$_id.gender",
      topColours: { $push: { colour: "$_id.baseColour", count: "$totalProductos" } }
    }
  },
  { $project: { _id: 0, gender: "$_id", topColours: { $slice: ["$topColours", 5] } } }
]);

//8. Agrupar los productos por año y montrar el número total de productos lanzados por cada año.
db.styles.aggregate([
  { $group: {
      _id: "$year",
      totalProductos: { $sum: 1 }
    }
  },
  { $sort: { _id: 1 } }, 
  { $project: { year: "$_id", totalProductos: 1, _id: 0 } }
]);

//9. Mostrar cómo se distribuyen los productos lanzados por año y estación
db.styles.aggregate([
  { $group: { 
      _id: { year: "$year", season: "$season" },
      totalProductos: { $sum: 1 }
    }
  },
  { $sort: { "_id.year": 1, "_id.season": 1 } }, 
  { $project: { 
      year: "$_id.year", 
      season: "$_id.season", 
      totalProductos: 1, 
      _id: 0 
    } 
  }
]);

//10. Mostrar el número de productos por género y las subcategorías más representativas de cada uno
db.styles.aggregate([
  { $match: { gender: { $in: ["Men", "Women"] } } }, 
  { $group: { 
      _id: { gender: "$gender", subCategory: "$subCategory" },
      totalProductos: { $sum: 1 }
    }
  },
  { $sort: { totalProductos: -1 } }, 
  { $group: { 
      _id: "$_id.gender", 
      subCategories: { 
        $push: { 
          subCategory: "$_id.subCategory", 
          totalProductos: "$totalProductos" 
        } 
      }, 
      totalProductos: { $sum: "$totalProductos" } 
    }
  },
  { $project: {
      _id: 1,
      totalProductos: 1,
      firstSubCategory: { $arrayElemAt: ["$subCategories.subCategory", 0] },
      secondSubCategory: { $arrayElemAt: ["$subCategories.subCategory", 1] }, 
      thirdSubCategory: { $arrayElemAt: ["$subCategories.subCategory", 2] },
    }
  },
  { $addFields: {
      description: { 
        $concat: [
          "En la categoría ", "$_id", 
          ", la subcategoría más popular es ", "$firstSubCategory", 
          ", seguida de ", "$secondSubCategory", 
          " y ", "$thirdSubCategory", "."
        ] 
      }
    }
  },
  { $project: { _id: 1, totalProductos: 1, description: 1 } }

//11. Mostrar los productos de la colección Primavera-Verano del año 2008 
//    para destacar las imágenes de estos mismos.
db.styles.aggregate([
    { $match: { season: { $in: ["Spring", "Summer"] } , year: 2008 } },
    { $project: {
        productDisplayName: 1,
        image_url: 1,
        season: 1
    }}
]);
