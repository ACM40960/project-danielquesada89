# Los pingüinos del Archipiélago de Palmer


El archipiélago Palmer es un grupo de islas ubicadas frente a la costa noroeste de la Península Antártica. Los pingüinos son especies recurrentes en estas islas. Por ello, se estudiará las características de la población de tres especies de pingüinos de la familia Pygoscelis anidadas en tres islas del Archipiélago de Palmer.



<p align="center">
  <img src="https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png" width="400">
</p>

## Objetivo


El objetivo es presentar un dashboard interactivo que permita estudiar las características de cada una de las especies que residen en las tres islas. Por ello, se podrán graficar cada una de las variables comparando los datos de las especies. Así como enfrentar cada una de las variables numéricas que forman el dataset y buscar un individuos en particular por su ID.

La app presenta una página de datos donde se muestra la base de datos y  las variables.


## Antes de ejecutar

### Contenidos del proyecto

El proyecto consta de tres carpetas:

* Limpieza: esta carpeta contiene el análisis exploratorio de los datos antes de subirlos a MongoDB. Asimismo, la creación de una variable location con las coordenadas de la isla en la que reside el pingüino.

* Api: contiene los códigos para la creación de la Api.

* Streamlit_multi: el código de creación del streamlit que representa los datos.



### Requirements
* [Requirements](https://github.com/teresababio/Project_Penguin/blob/main/requirements.txt)
### Cómo se ejecuta el proyecto

Para ejecutar la api se tiene que abrir la terminal y ejecutar:

```bash
cd api
./run_api.sh
```

Una vez hecho esto, se abre otra terminal y se ejecuta lo siguiente para activar el dashboard.

```bash
cd streamlit_multi
./run_streamlit.sh
```
