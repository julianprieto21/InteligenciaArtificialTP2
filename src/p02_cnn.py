import numpy as np
import matplotlib.pyplot as plt
import math

MAX_POOL_SIZE = 5
CONVOLUTION_SIZE = 4
CONVOLUTION_FILTERS = 2


def get_initial_params():
    """
    Calcule los parámetros iniciales para la red neuronal.

    Esta función debe devolver un diccionario mappearno los nombres de parámetros a arrays de numpy que contienen
    los valores iniciales de esos parámetros.

    Debe haber cuatro parámetros para este modelo:
    W1 es la matriz de peso para la capa convolucional
    b1 es el vector bias para la capa convolucional
    W2 es la matriz de peso para las capas de salida
    b2 es el vector bias para la capa de salida

    Las matrices de peso deben inicializarse con valores extraídos de una distribución normal aleatoria.
    La media de esa distribución debe ser 0.
    La varianza de esa distribución debe ser 1/sqrt(n) donde n es el número de neuronas que alimenta una salida para esa capa.

    Los vectores de bias deben inicializarse con cero.


    Return:
         Un dict mapeando nombres de parámetros a arrays numpy
    """

    size_after_convolution = 28 - CONVOLUTION_SIZE + 1
    size_after_max_pooling = size_after_convolution // MAX_POOL_SIZE

    num_hidden = size_after_max_pooling * size_after_max_pooling * CONVOLUTION_FILTERS

    return {
        "W1": np.random.normal(
            size=(CONVOLUTION_FILTERS, 1, CONVOLUTION_SIZE, CONVOLUTION_SIZE),
            scale=1 / math.sqrt(CONVOLUTION_SIZE * CONVOLUTION_SIZE),
        ),
        "b1": np.zeros(CONVOLUTION_FILTERS),
        "W2": np.random.normal(size=(num_hidden, 10), scale=1 / math.sqrt(num_hidden)),
        "b2": np.zeros(10),
    }


def forward_convolution(conv_W, conv_b, data):
    """
    Calcula la salida de una capa convolucional dados los pesos y los datos.

    conv_W tiene tamaño (# canales de salida, # canales de entrada, ancho de convolución, altura de convolución)
    conv_b tiene tamaño (# canales de salida)

    los datos son de la forma (# canales de entrada, ancho, alto)

    La salida debe ser el resultado de una convolución y debe tener el tamaño:
        (# canales de salida, ancho - ancho de convolución + 1, altura - altura de convolución + 1)

    Returns:
        La salida de la convolución como array numpy
    """

    conv_channels, _, conv_width, conv_height = conv_W.shape

    input_channels, input_width, input_height = data.shape

    output = np.zeros(
        (conv_channels, input_width - conv_width + 1, input_height - conv_height + 1)
    )

    for x in range(input_width - conv_width + 1):
        for y in range(input_height - conv_height + 1):
            for output_channel in range(conv_channels):
                output[output_channel, x, y] = (
                    np.sum(
                        np.multiply(
                            data[:, x : (x + conv_width), y : (y + conv_height)],
                            conv_W[output_channel, :, :, :],
                        )
                    )
                    + conv_b[output_channel]
                )

    return output


def backward_convolution(conv_W, conv_b, data, output_grad):
    """
    Calcula el gradiente de la pérdida con respecto a los parámetros de la convolución.

    Ver forward_convolution para los tamaños de las entradas.
    output_grad es el gradiente de la pérdida con respecto a la salida de la convolución.

    Returns:
        Una tupla que contiene 3 gradientes.
        El primer elemento es el gradiente de la pérdida con respecto a los pesos de convolución.
        El segundo elemento es el gradiente de la pérdida con respecto al bias de convolución.
        El tercer elemento es el gradiente de la pérdida con respecto a los datos de entrada.
    """


# *** EMPEZAR CÓDIGO AQUÍ ***

    conv_channels, _, conv_width, conv_height = conv_W.shape
    input_channels, input_width, input_height = data.shape
    # output_width, output_height = output_grad.shape[1], output_grad.shape[2]

    grad_W = np.zeros_like(conv_W)
    grad_b = np.zeros_like(conv_b)
    grad_x = np.zeros_like(data)

    for x in range(input_width - conv_width + 1):#range(output_width):
        for y in range(input_height - conv_height + 1):#range(output_height):
            for output_channel in range(conv_channels):
                # Gradiente con respecto a pesos
                grad_W[output_channel, :, :, :] += data[:, x : (x + conv_width), y : (y + conv_height)] * output_grad[output_channel, x, y]
                # Gradiente con respecto a los bias
                grad_b[output_channel] += output_grad[output_channel, x, y]
                # Gradiente con respecto a X´s
                grad_x[:, x : (x + conv_width), y : (y + conv_height)] += conv_W[output_channel, :, :, :] * output_grad[output_channel, x, y]

    return grad_W, grad_b, grad_x

# *** TERMINAR CÓDIGO AQUÍ ***


def forward_max_pool(data, pool_width, pool_height):
    """
    Calcula la salida de una capa max pooling dados los datos y las dimensiones de la grilla.

    La longitud del stride debe ser igual al tamaño del pool.

    los datos son de la forma (# canales, ancho, alto)

    La salida debe ser el resultado de la capa max pooling y debe tener el tamaño:
        (# canales, ancho // pool_width, altura // pool_height)

    Returns:
        El resultado de la capa max pooling
    """
    input_channels, input_width, input_height = data.shape

    output = np.zeros(
        (input_channels, input_width // pool_width, input_height // pool_height)
    )

    for x in range(0, input_width, pool_width):
        for y in range(0, input_height, pool_height):
            output[:, x // pool_width, y // pool_height] = np.amax(
                data[:, x : (x + pool_width), y : (y + pool_height)], axis=(1, 2)
            )

    return output


def backward_max_pool(data, pool_width, pool_height, output_grad):
    """
    Calcule el gradiente de la pérdida con respecto a los datos en la capa max pooling.

    los datos son de la forma (# canales, ancho, alto)
    output_grad tiene forma (# canales, ancho // pool_width, alto // pool_height)

    output_grad es el gradiente de la pérdida con respecto a la salida del backward de la capa max pooling

    Returns:
        El gradiente de la pérdida con respecto a los datos (mismo tamaño que los datos)
    """

    # *** EMPEZAR CÓDIGO AQUÍ ***

    output_width, output_height = output_grad.shape[1], output_grad.shape[2]
    input_channels, input_width, input_height = data.shape
    grad_data = np.zeros((CONVOLUTION_FILTERS, output_width * 5, output_height * 5))

    for x in range(0, input_width - pool_width, pool_width):
        for y in range(0, input_height - pool_height, pool_height):
            for channel in range(CONVOLUTION_FILTERS):
                patch = data[:, x : (x + pool_width), y : (y + pool_height)]
                # print(patch, patch.shape)
                mask = (patch == np.max(patch))
                # print(mask, mask.shape)
                # print(output_grad[channel, x // pool_width, y // pool_height], output_grad[channel, x // pool_width, y // pool_height].shape)
                grad_data[channel, x : (x + pool_width), y : (y + pool_height)] = output_grad[channel, x // pool_width, y // pool_height] * mask
                # print(grad_data[channel, x : (x + pool_width), y : (y + pool_height)], grad_data[channel, x : (x + pool_width), y : (y + pool_height)].shape)

    return grad_data

    # *** TERMINAR CÓDIGO AQUÍ ***


def forward_relu(x):
    """
    Calcula la ReLU para x.

    Args:
        x: un vector floats numpy

    Return:
        un vector floats numpy con el resultado de ReLU
    """
    x[x <= 0] = 0

    return x


def backward_relu(x, grad_outputs):
    """
    Calcula el gradiente de la pérdida resp a x

    Args:
        x: un array de numpy de tamaño arbitrario.
        grad_outputs: un array de numpy del mismo tamaño que x que contiene el gradiente de la pérdida con respecto
            a la salida de relu

    Return:
        Un array numpy del mismo tamaño que x que contiene los gradientes con respecto a x.
    """

    # *** EMPEZAR CÓDIGO AQUÍ ***

    grad = np.zeros_like(x)
    grad[x > 0] = grad_outputs[x > 0]
    return grad
 
    # *** TERMINAR CÓDIGO AQUÍ ***


def forward_linear(weights, bias, data):
    """
    Calcule la salida de una capa lineal con los pesos, el bias y los datos proporcionados.
    pesos de tamaño (# características de entrada, # características de de salida)
    el bias de tamaño (# características de de salida)
    los datos de tamaño (# características de entrada)

    La salida debe tener tamaño (# características de de salida)

    Returns:
        El resultado de la capa lineal.
    """
    return data.dot(weights) + bias


def backward_linear(weights, bias, data, output_grad):
    """
    Calcula los gradientes de pérdida con respecto a los parámetros de una capa lineal.

    Consulte forward_linear para obtener información sobre los tamaños de las variables.

    output_grad es el gradiente de la pérdida con respecto a la salida de esta capa.

    Esto debería devolver una tupla con tres elementos:
    - El gradiente de la pérdida con respecto a los pesos
    - El gradiente de la pérdida con respecto al bias
    - El gradiente de la pérdida con respecto a los datos
    """

    # *** EMPEZAR CÓDIGO AQUÍ ***
    grad_weights = data.T.dot(output_grad)
    grad_bias =  output_grad #np.sum(output_grad, axis=0)
    grad_data = output_grad.dot(weights.T)

    return [grad_weights, grad_bias, grad_data]

    # *** TERMINAR CÓDIGO AQUÍ ***

def forward_softmax(x):
    """
    Calcula la función softmax para un solo ejemplo.
    El tamaño de la entrada es # clases.

    Nota importante: debe tener cuidado para evitar el overflow de esta función. Funciones
    como softmax tienen tendencia a overflow cuando se calculan números muy grandes como e^10000.
    Sabrá que su función es resistente al overflow cuando puede manejar entradas como:
    np.array([[10000, 10010, 10]]) sin problemas.

        x: un array de floats numpy 1d de tamaño #clases

    Salida:
        un array de floats numpy 1d  que contiene los resultados de softmax.
    """
    x = x - np.max(x, axis=0)
    exp = np.exp(x)
    s = exp / np.sum(exp, axis=0)
    return s


def backward_softmax(x, grad_outputs):
    """
    Calcule el gradiente de la pérdida con respecto a x.

    grad_outputs es el gradiente de la pérdida con respecto a las salidas del softmax.

    Argumentos:
        x: un vector floats numpy 1d de tamaño #clases
        grad_outputs: un vector floats numpy 1d de de tamaño #clases

    Salida:
        un vector floats numpy 1d de la misma forma que x con la derivada de la pérdida con respecto a x
    """

    # *** EMPEZAR CÓDIGO AQUÍ ***

    softmax_x = forward_softmax(x)
    grad = softmax_x * (grad_outputs - np.dot(grad_outputs, softmax_x))
    # return grad
    return grad_outputs * (softmax_x * (1-softmax_x)) # Mismo calculo que arriba

    # *** TERMINAR CÓDIGO AQUÍ ***


def forward_cross_entropy_loss(probabilities, labels):
    """
    Calcule la salida de una capa de entropía cruzada dadas las probabilidades y las etiquetas.

    probabilidades con tamaño (# clases)
    etiquetas con tamaño (# clases)

    La salida debe ser un escalar.

    Returns:
        El resultado log pérdida de la capa
    """

    result = 0

    for i, label in enumerate(labels):
        if label == 1:
            result += -np.log(probabilities[i])

    return result


def backward_cross_entropy_loss(probabilities, labels):
    """
    Calcule el gradiente de la entropía cruzada con respecto a las probabilidades.

    probabilidades con tamaño (# clases)
    etiquetas con tamaño (# clases)

    La salida debe ser el gradiente con respecto a las probabilidades.

    Returns:
        El gradiente de la pérdida con respecto a las probabilidades.
    """

    # *** EMPEZAR CÓDIGO AQUÍ ***

    # Introducir una pequeña constante para evitar divisiones por cero
    # epsilon = 1e-10

    # Utilizar np.clip para evitar que las probabilidades sean extremadamente cercanas a cero
    # probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    # print(probabilities)
    print(probabilities)
    grad = -labels / probabilities
    return grad

    # *** TERMINAR CÓDIGO AQUÍ ***


def forward_prop(data, labels, params):
    """
    Implemente la pasada foward dados los datos, las etiquetas y los parámetros.

    Argumentos:
        datos: un numpy array que contiene la entrada (tamaño 1 por 28 por 28)
        etiquetas: un numpy array 1d que contiene las etiquetas (tamaño 10)
        params: un diccionario que mapea nombres de parámetros a arrays numpy con los parámetros.
            Esta matriz numpy contendrá W1, b1, W2 y b2
            W1 y b1 representan los pesos y bias de la capa oculta de la red
            W2 y b2 representan los pesos y el bias de la capa de salida de la red

    Returns:
        Una tupla de 2 elementos que contiene:
            1. Un array numpy: La salida (después del softmax) de la capa de salida
            2. La pérdida promedio para estos datos
    """

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    first_convolution = forward_convolution(W1, b1, data)
    first_max_pool = forward_max_pool(first_convolution, MAX_POOL_SIZE, MAX_POOL_SIZE)
    first_after_relu = forward_relu(first_max_pool)

    flattened = np.reshape(first_after_relu, (-1))

    logits = forward_linear(W2, b2, flattened)

    y = forward_softmax(logits)
    cost = forward_cross_entropy_loss(y, labels)

    return y, cost


def backward_prop(data, labels, params):
    """
    Implementar el cálculo del gradiente para una red neuronal. Es decir la pasada backward completa.

    Argumentos:
        datos: un array numpy que contiene la entrada para un solo ejemplo
        etiquetas: un array numpy 1d que contiene las etiquetas para un solo ejemplo
        params: un diccionario que mapea nombres de parámetros a arrays numpy con los parámetros.
            Esta matriz numpy contendrá W1, b1, W2 y b2
            W1 y b1 representan los pesos y el bias de la capa convolucional
            W2 y b2 representan los pesos y el bias de la capa de salida de la red

    Returns:
        Un diccionario de strings a arrays de numpy donde cada clave representa el nombre de un peso
        y los valores representan el gradiente de la pérdida con respecto a ese peso.

        En particular, debe tener 4 elementos:
            W1, W2, b1 y b2
    """

    # *** EMPEZAR CÓDIGO AQUÍ ***

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    y, cost = forward_prop(data, labels, params)
    grad_softmax = backward_cross_entropy_loss(y, labels)
    # print(grad_softmax.shape)
    # print(grad_softmax)
    grad_logits = backward_softmax(y, grad_softmax)
    # print(grad_logits.shape)
    # print(grad_logits)
    grad_W2, grad_b2, grad_relu = backward_linear(W2, b2, y, grad_logits)
    # print(grad_relu.shape)
    # print(grad_relu)
    grad_max_pool = backward_relu(grad_relu, grad_relu).reshape(CONVOLUTION_FILTERS, MAX_POOL_SIZE, MAX_POOL_SIZE)
    # print(grad_max_pool.shape)
    # print(grad_max_pool)
    grad_convolution = backward_max_pool(data, MAX_POOL_SIZE, MAX_POOL_SIZE, grad_max_pool)
    # print(grad_convolution.shape)
    # print(grad_convolution)
    grad_W1, grad_b1, grad_data = backward_convolution(W1, b1, data, grad_convolution)
    # print(grad_data)

    gradientes = {
        'W1': grad_W1,
        'b1': grad_b1,
        'W2': grad_W2,
        'b2': grad_b2
    }

    return gradientes

    # *** TERMINAR CÓDIGO AQUÍ ***


def forward_prop_batch(batch_data, batch_labels, params, forward_prop_func):
    """Aplique prop foward a cada imagen en un lote"""

    y_array = []
    cost_array = []

    for item, label in zip(batch_data, batch_labels):
        y, cost = forward_prop_func(item, label, params)
        y_array.append(y)
        cost_array.append(cost)

    return np.array(y_array), np.array(cost_array)


def gradient_descent_batch(
    batch_data, batch_labels, learning_rate, params, backward_prop_func
):
    """
    Realice un descenso por gradiente de un lote de datos de entrenamiento proporcionados,
    utilizando la tasa de aprendizaje proporcionada.

    Este código debería actualizar los parámetros almacenados en params.
    No debe devolver nada

    Argumentos:
        batch_data: un array numpy que contiene los datos de entrenamiento para el lote
        train_labels: un array numpy que contiene las etiquetas de entrenamiento para el lote
        learning_rate: La tasa de aprendizaje
        params: un dict de nombres de parámetros a valores de parámetros que deben actualizarse.
        backwards_prop_func: una función que sigue API de backwards_prop

    Returns: esta función no devuelve nada.
    """

    total_grad = {}

    for i in range(batch_data.shape[0]):
        # plt.imshow(batch_data[i, :].reshape(28,28),cmap="gray")
        # plt.colorbar()
        # plt.title("Label: " + str(batch_labels[i,:]))
        # plt.show()
        grad = backward_prop_func(batch_data[i, :, :], batch_labels[i, :], params)
        for key, value in grad.items():
            if key not in total_grad:
                total_grad[key] = np.zeros(value.shape)

            total_grad[key] += value

    params["W1"] = params["W1"] - learning_rate * total_grad["W1"]
    params["W2"] = params["W2"] - learning_rate * total_grad["W2"]
    params["b1"] = params["b1"] - learning_rate * total_grad["b1"]
    params["b2"] = params["b2"] - learning_rate * total_grad["b2"]

    # Esta funcióno no devuelve nada
    return


def nn_train(
    train_data,
    train_labels,
    dev_data,
    dev_labels,
    get_initial_params_func,
    forward_prop_func,
    backward_prop_func,
    learning_rate=5.0,
    batch_size=16,
    num_batches=400,
):
    m = train_data.shape[0]

    params = get_initial_params_func()

    cost_dev = []
    accuracy_dev = []
    for batch in range(num_batches):

        batch_data = train_data[batch * batch_size : (batch + 1) * batch_size, :, :, :]
        batch_labels = train_labels[batch * batch_size : (batch + 1) * batch_size, :]

        if batch % 100 == 0:
            output, cost = forward_prop_batch(
                dev_data, dev_labels, params, forward_prop_func
            )
            print(sum(cost), len(cost))
            cost_dev.append(sum(cost) / len(cost))
            accuracy_dev.append(compute_accuracy(output, dev_labels))

            print("Costo y accuracy", cost_dev[-1], accuracy_dev[-1])

        gradient_descent_batch(
            batch_data, batch_labels, learning_rate, params, backward_prop_func
        )

    return params, cost_dev, accuracy_dev


def nn_test(data, labels, params):
    output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    correct_output = np.argmax(output, axis=1)
    correct_labels = np.argmax(labels, axis=1)

    is_correct = [a == b for a, b in zip(correct_output, correct_labels)]

    accuracy = sum(is_correct) * 1.0 / labels.shape[0]
    return accuracy


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=",")
    y = np.loadtxt(labels_file, delimiter=",")

    x = np.reshape(x, (x.shape[0], 1, 28, 28))

    return x, y


def run_train(all_data, all_labels, backward_prop_func):
    params, cost_dev, accuracy_dev = nn_train(
        all_data["train"],
        all_labels["train"],
        all_data["dev"],
        all_labels["dev"],
        get_initial_params,
        forward_prop,
        backward_prop_func,
        learning_rate=1e-2,
        batch_size=16,
        num_batches=400,
    )

    # t = np.arange(400 // 100)

    # fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax1.plot(t, cost_dev, "b")
    # ax1.set_xlabel("tiempo")
    # ax1.set_ylabel("pérdida")
    # ax1.set_title("Curva train")

    # ax2.plot(t, accuracy_dev, "b")
    # ax2.set_xlabel("tiempo")
    # ax2.set_ylabel("accuracy")

    # fig.savefig("output/train.png")


def main():
    np.random.seed(100)
    train_data, train_labels = read_data(
        "./data/images_train.csv", "./data/labels_train.csv"
    )
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p, :]
    train_labels = train_labels[p, :]

    dev_data = train_data[0:400, :]
    dev_labels = train_labels[0:400, :]
    train_data = train_data[400:, :]
    train_labels = train_labels[400:, :]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    all_data = {
        "train": train_data,
        "dev": dev_data,
    }

    all_labels = {
        "train": train_labels,
        "dev": dev_labels,
    }

    run_train(all_data, all_labels, backward_prop)


if __name__ == "__main__":
    main()