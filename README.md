# **Movies Challenge**

El siguiente proyecto ha sido desarrollado usando un ambiente limpio (tipo `pyenv`) que tiene como base python `3.9.7`.

Contiene tres jupyter-notebook principales:
* __`movies_challenge_preproccessing.ipynb`__: En él se explica cómo se ha llevado a cabo la ingeniería de características.
* __`movies_challenge_training_optimization_features.ipynb`__: En él se lleva a cabo el entrenamiento y selección de hiperparámetros.

* __`movies_challenge_feature_importances_and_the_end.ipynb`__: En él se exhibe qué características tienen mayor impacto en las predicciones que hace el modelo. Se exhibe un resumen del trabajo además de comentarios sobre qué se pudiese mejorar si se contara con más tiempo.

La paqueterías necesarias a instalar están contenidas en `requirements.txt`.

Se supone en todo momento que los archivos
* `genome_scores.csv`,
* `genome_tags.csv`,
* `link.csv`,
* `movie.csv`
* `rating.csv`

ya están contenidos en la carpeta *movies_challenge*.

Los archivos "formales", es decir, en formato limpio y funcionales son:
* `extras.py`
* `cluster.py`
* `past_information_user.py`
* `feature_engineering_user.py`
* `EDA_functions.py`.

Todos los demás archivos con terminación `.py` están "en sucio" pero se incluyen para no perder la etapa de experimentación que se ha llevado a cabo en ellos.

Es menester incluir en la carpeta raíz los archivos contenidos en __"Archivos necesarios.zip"__ que se ha enviado por correo electrónico para poder reproducir los jupyter-notebook principales.
