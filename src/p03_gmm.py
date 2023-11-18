import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors para los gráficos
K = 4           # cantida de gaussianas mezcladas
NUM_TRIALS = 1  # cantidad de corridas (se puede modificar para debugging)
UNLABELED = -1  # etiqueta para los datos no etiquetados (no cambiar)


def main(is_semi_supervised, trial_num):
    print('Corriendo EM {} ...'
          .format('semi-supervisado' if is_semi_supervised else 'no supervisado'))

    # Cargar dataset
    train_path = os.path.join('.', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Dividir en ejemplos etiquetados y no etiquetados
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # ejemplos etiquetados
        z = z[labeled_idxs, :]         # etiquetas correspondientes
        x = x[~labeled_idxs, :]        # ejemplos no etiquetados

    # *** EMPEZAR CÓDIGO AQUÍ ***
    # (1) Inicialice mu y sigma dividiendo los m datos uniformemente al azar
    # en K grupos, luego calculando la media de la muestra y la covarianza para cada grupo
    mu = np.zeros((K, x.shape[1]))
    sigma = np.zeros((K, x.shape[1], x.shape[1]))
    for i in range(K):
        idx = np.random.choice(x.shape[0], size=x.shape[0] // K, replace=False) # Elijo una muestra al azar de mis datos
        mu[i] = np.mean(x[idx], axis=0) # Calculo la media
        sigma[i] = np.cov(x[idx].T) # Y la covarianza

    # (2) Inicialice phi para colocar la misma probabilidad en cada gaussiana
    # phi debe ser un array numpy de tamaño (K,)
    phi = np.ones(K) / K

    # (3) Inicialice los valores de w para colocar la misma probabilidad en cada gaussiana
    # w debe ser un array numpy de tamaño (m, K)
    w = np.zeros((x.shape[0], K))
    for i in range(x.shape[0]):
        w[i, :] = phi

    # print(mu.shape)
    # print(sigma.shape)
    # print(phi.shape)
    # print(w.shape)

    # *** TERMINAR CÓDIGO AQUÍ ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # graficar predicciones
    # z_pred = np.zeros(m)
    # if w is not None:  # solamente para código de inicio. Cambiar según corresponda.
    #     for i in range(m):
    #         z_pred[i] = np.argmax(w[i])

    # plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """
    Consulte los comentarios entre líneas para obtener más información.

    Argumentos:
        x: Matriz de diseño de tamaño (m, n).
        w: Matriz de pesos inicial de tamaño (m, k).
        phi: prior inicial para la mezcla, de tamaño (k,).
        mu: medias iniciales de los clusters, lista de k arrays de tamaño (n,).
        sigma: covarianzas iniciales de clusters, lista de k matrices de tamaño (n, n).

    Returns:
        Matriz de peso actualizada de tamaño (m, k) resultante del algoritmo EM.
        Más específicamente, w[i, j] debe contener la probabilidad de que
        el ejemplo x^(i) pertenezca a la j-ésima gaussiana de la mezcla.
    """
    # No es necesario cambiar ninguno de estos parámetros.
    eps = 1e-3  # umbral de convergencia
    max_iter = 1000
    m = x.shape[1]

    # parar cuando el cambio absoluto en log-verosimilitud sea < eps.
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # pass  # solamente para código inicial. Cambiar según corresponda.
        # *** EMPEZAR CÓDIGO AQUÍ ***
        # (1) E-step: actualice sus estimaciones en w
        

        # (2) M-step: actualice los parámetros del modelo phi, mu y sigma
        phi = 1/m * np.sum(w, axis=0)
        print(x.shape)
        print(w.shape)
        mu = (w.T @ x) / w
        print(mu.shape)
        sigma = np.sum(w * np.outer(x - mu, x - mu), axis=0) / np.sum(w, axis=0) # ?

        # (3) Calcule la log probabilidad (log likelihood = ll) de los datos para verificar la convergencia.
        # Por log-verosimilitud, nos referimos a `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # Definimos la convergencia por abs(ll - prev_ll) < eps.
        # Sugerencia: para debugging, recuerde que ll debería ser monótonamente creciente.
        

        # *** TERMINAR CÓDIGO AQUÍ ***

    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """
    Mirar los comentarios entre líneas para obtener más información.

    Argumentos:
        x: Matriz de ejemplos no etiquetados de tamaño (m, n).
        x_tilde: matriz de ejemplos etiquetados de tamaño (m_tilde, n).
        z: Array de etiquetas de tamaño (m_tilde, 1).
        w: Matriz de pesos inicial de tamaño (m, k).
        phi: prior inicial de la mezcla, de tamaño (k,).
        mu: medias iniciales de los clusters, lista de k arrays de tamaño (n,).
        sigma: covarianzas iniciales de los clusters, lista de k matrices de tamaño (n, n).

    Returns:
        Matriz de pesos actualizada de tamaño (m, k) resultante del algoritmo EM semisupervisado.
        Más específicamente, w[i, j] debe contener la probabilidad de que
        el ejemplo x^(i) pertenezca a la j-ésima Gaussiana de la mezcla.
    """
    # No es necesario cambiar ninguno de estos parámetros.
    alpha = 20.  # Peso para los ejemplos etiquetados
    eps = 1e-3   # umbral de convergencia
    max_iter = 1000

    # parar cuando el cambio absoluto en log-verosimilitud sea < eps.
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # solamente para código inicial. Cambiar según corresponda.
        # *** EMPEZAR CÓDIGO AQUÍ ***
        # (1) E-step: actualice sus estimaciones en w
        # (2) M-step: actualice los parámetros del modelo phi, mu y sigma
        # (3) Calcule la log probabilidad (log likelihood = ll) de los datos para verificar la convergencia.
        # Sugerencia: asegúrese de incluir alfa en su cálculo de ll.
        # Sugerencia: para debugging, recuerde que ll debería ser monótonamente creciente.
        # *** TERMINAR CÓDIGO AQUÍ ***

    return w


# *** EMPEZAR CÓDIGO AQUÍ ***
# funciones helpers (utilidades) que desee
# *** TERMINAR CÓDIGO AQUÍ ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Grafica predicciones de GMM de un conjunto de datos 2D `x` con etiquetas `z`.

    Escribe en el directorio de salida, incluyendo `plot_id`
    en el nombre, y agregando 'ss' si el GMM era supervisado.

    NOTA: No necesita editar esta función.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM predicciones'.format('Semi-supervisado' if with_supervision else 'No supervisado'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Cargue el conjunto de datos para el modelo de mezcla gaussiana.

    Argumentos:
        csv_path: ruta al archivo CSV que contiene el conjunto de datos.

    Returns:
         x: array NumPy (m, n)
         z: array NumPy (m, 1)

    NOTA: No necesita editar esta función.
    """

    # cargar headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # cargar features y etiquetas
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Ejecute NUM_TRIALS para ver cómo las diferentes inicializaciones
    # afectan las predicciones finales con y sin supervisión
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** EMPEZAR CÓDIGO AQUÍ ***
        # Una vez que haya implementado la versión semi-supervisada,
        # descomente la siguiente línea.
        # No necesita agregar ninguna otra línea en este bloque de código.
        # main(with_supervision=True, trial_num=t)
        # *** TERMINAR CÓDIGO AQUÍ ***
