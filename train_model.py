import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def main():
    # Carregamento e pre-processamento do dataset MNIST
    print("Carregando dataset MNIST...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalização dos valores dos píxeis para o intervalo [0, 1] e ajuste de dimensões
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    print(f"Dataset carregado. Treino: {x_train.shape[0]} imagens. Teste: {x_test.shape[0]} imagens.\n")

    # Construção da CNN
    print("Construindo a arquitetura do modelo...")
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.summary()

    #  Compilação e Treino do Modelo
    print("\nCompilando o modelo...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nIniciando o treianmento (5 épocas)...")
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    # Avaliação e Extração de Múltiplas Métricas
    print("\nAvaliando o modelo com os dados de teste...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("\n" + "="*60)
    print("RESULTADOS DA AVALIAÇÃO")
    print("="*60)
    print(f"-> Accuracy (Acurácia): {test_acc*100:.2f}%")
    print(f"-> Loss (Perda):        {test_loss:.4f}")
    print("-" * 60)
    print("[Nota sobre o modelo]")
    print("A rede conseguiu manter uma acurácia excelente mesmo com uma arquitetura mais ismples. Além disso, o loss baixo mostra que  o modelo tem confiança nas previsões.\n")
    print("Como a ideia é rodar esse modelo em hardware com recursos limitados, esse balanço entre alta precisão e baixo custo computacional é ótimo.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()