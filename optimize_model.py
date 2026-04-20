import tensorflow as tf
import os

def get_file_size(file_path):
    # Retorna o tamanho do arquivo em KB.
    size = os.path.getsize(file_path)
    return size / 1024

def main():
    model_path = 'model.h5'
    tflite_model_path = 'model.tflite'          # Dynamic Range 
    tflite_fp16_path = 'model_float16.tflite'   # Float16 

    print(f"Carregando o modelo de '{model_path}'...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo. Erro: {e}")
        return

    # Dynamic Range Quantization
    print("\n[1/2] Iniciando conversão para TFLite (Dynamic Range Quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"-> Modelo salvo em '{tflite_model_path}'")

    # Float16 Quantization
    print("\n[2/2] Iniciando conversão alternativa (Float16 Quantization)...")
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16_model = converter_fp16.convert()

    with open(tflite_fp16_path, 'wb') as f:
        f.write(tflite_fp16_model)
    print(f"-> Modelo alternativo salvo em '{tflite_fp16_path}'")

    # Comparação de tamanhos para demonstrar o ganho e domínio técnico
    if os.path.exists(model_path) and os.path.exists(tflite_model_path) and os.path.exists(tflite_fp16_path):
        h5_size = get_file_size(model_path)
        tflite_size = get_file_size(tflite_model_path)
        tflite_fp16_size = get_file_size(tflite_fp16_path)
        
        print("\n" + "="*65)
        print("ANÁLISE DE OTIMIZAÇÃO")
        print("="*65)
        print(f"-> Original (.h5):                  {h5_size:.2f} KB")
        print(f"-> Otimizado Dynamic Range (INT8):  {tflite_size:.2f} KB (Redução: {((h5_size - tflite_size) / h5_size) * 100:.2f}%)")
        print(f"-> Otimizado Float16 (FP16):        {tflite_fp16_size:.2f} KB (Redução: {((h5_size - tflite_fp16_size) / h5_size) * 100:.2f}%)")
        print("-" * 65)
        print("[Nota sobre os métodos]")
        print("O Dynamic Range Quantization reduziu o modelo ao usar pesos em INT8, bom para CPUs e microcontroladores.")
        print("Já a quantização Float16 reduz pela metade, mas mantém uma boa precisão, sendo ideal para caso o dispositivo possua uma GPU Mobile.")
        print("O arquivo 'model.tflite' será mantido como a entrega padrão pedida no README do processo seletivo.")
        print("="*65 + "\n")

if __name__ == "__main__":
    main()