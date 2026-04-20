import tensorflow as tf
import os

def get_file_size(file_path):
    #Retorna o tamanho do arquivo em KB.
    size = os.path.getsize(file_path)
    return size / 1024

def main():
    model_path = 'model.h5'
    tflite_model_path = 'model.tflite'

    print(f"Carregando o modelo original de '{model_path}'...")
    # Carrega o modelo treinado 
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo. Certifique-se de que o train_model.py foi executado. Erro: {e}")
        return

    print("\nIniciando conversão para TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Aplica a Dynamic Range Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("Aplicando Dynamic Range Quantization...")
    tflite_model = converter.convert()

    # Salva o modelo otimizado no disco
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\nModelo TFLite salvo com sucesso em '{tflite_model_path}'")

    # Comparação de tamanhos para demonstrar o ganho da otimização
    if os.path.exists(model_path) and os.path.exists(tflite_model_path):
        h5_size = get_file_size(model_path)
        tflite_size = get_file_size(tflite_model_path)
        
        print("\n" + "="*60)
        print("ANÁLISE DE OTIMIZAÇÃO")
        print("="*60)
        print(f"-> Tamanho do modelo original (.h5):     {h5_size:.2f} KB")
        print(f"-> Tamanho do modelo otimizado (.tflite): {tflite_size:.2f} KB")
        print(f"-> Redução de tamanho:                    {((h5_size - tflite_size) / h5_size) * 100:.2f}%")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()