import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def main():
    # Carregamento e pre-processamento do dataset MNIST
    print("Iniciando o carregamento do dataset MNIST...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalização dos valores dos pixels para o intervalo [0, 1] e ajuste de dimensões
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    print(f"Dataset carregado! Treino: {x_train.shape[0]} imagens. Teste: {x_test.shape[0]} imagens.")

    # Construção da Arquitetura da CNN
    print("Construindo a arquitetura do modelo...")
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Achatamento e camadas densas
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax') # 10 classes de saída (dígitos de 0 a 9)
    ])

    # Exibe um resumo da arquitetura no terminal
    model.summary()

if __name__ == "__main__":
    main()