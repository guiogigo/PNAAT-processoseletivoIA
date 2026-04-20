# Processo Seletivo – Intensivo Maker | AI

**👤 Identificação:** Guilherme Viana Batista

---

## 1️⃣ Resumo da Arquitetura do Modelo

Para garantir máxima eficiência em dispositivos Edge que possuem recurso limitados, projetei uma Rede Neural Convolucional (CNN) no arquivo `train_model.py`. A arquitetura é composta por:
* **Camadas de Extração de Características:** 
  * 1x Conv2D (16 filtros, 3x3, ReLU) + MaxPooling2D (2x2)
  * 1x Conv2D (32 filtros, 3x3, ReLU) + MaxPooling2D (2x2)
* **Camadas de Classificação:**
  * Flatten (Para vetorizar os mapas de características)
  * Dense (32 neurônios, ReLU)
  * Dense de Saída (10 neurônios, Softmax para classificação dos dígitos de 0 a 9)

Essa estrutura evita o uso de redes profundas desnecessárias, priorizando a extração rápida de padrões básicos dos dígitos do MNIST.

## 2️⃣ Bibliotecas Utilizadas

* **TensorFlow / Keras (v2.x):** Para construção, treinamento e conversão do modelo de aprendizado.
* **OS:** Biblioteca nativa do Python para manipulação de caminhos de arquivos e cálculo do tamanho dos modelos em disco.

## 3️⃣ Técnica de Otimização do Modelo

No arquivo `optimize_model.py`, apliquei e comparei duas técnicas distintas de otimização para Edge AI:

1. **Dynamic Range Quantization (Principal):** Converte os pesos do modelo de Float32 para inteiros de 8 bits (INT8) durante a inferência. É uma boa técnica para microcontroladores e CPUs de baixo custo, pois reduz o tamanho do modelo em quase 4x com perda mínima de acurácia. O modelo final foi salvo como `model.tflite`.
2. **Float16 Quantization:** Reduz os pesos de Float32 para Float16. Apesar de a redução de tamanho ser menor, mantém maior precisão e é uma boa escolha caso o dispositivo Edge possuir uma GPU Mobile ou acelerador de hardware (NPU). O modelo foi salvo como `model_float16.tflite`.

## 4️⃣ Resultados Obtidos

* **Treinamento:** O modelo atingiu uma excelente Acurácia, geralmente superior a 98% no conjunto de testes, e um Loss muito baixo após apenas 5 épocas, comprovando a eficácia da arquitetura.
* **Otimização:** 
  * Modelo Original (.h5): ~150 a 200 KB
  * Otimizado pelo Float16: Redução de ~50%
  * Otimizado pelo Dynamic Range: Redução de ~70 a 80%.

*(Nota: Os valores exatos podem ser verificados nos logs do terminal durante a execução).*

## 5️⃣ Comentários Adicionais

* **Decisões Técnicas Importantes:** A principal decisão foi limitar o número de filtros convolucionais (16 e 32) e os neurônios da camada densa (32). Redes tradicionais para MNIST usam 64/128 filtros, mas para o contexto de **Edge AI**, isso desperdiçaria memória RAM e bateria do dispositivo embarcado sem ganhos perceptíveis de acurácia.
* **Aprendizados:** O desafio consolidou o entendimento do *trade-off* entre tamanho do modelo vs. capacidade de processamento, provando que o trabalho do Engenheiro de IA vai muito além do `model.fit()`, exigindo adequação ao hardware final.

* **Dificuldades encontradas:**
  * **Avisos de Depreciação (Warnings) do TensorFlow:** Lidar com os avisos de formato legado ao salvar o modelo em .h5, pois o TF agora recomenda o .keras. Tive que tomar a decisão consciente de manter o .h5 para garantir a compatibilidade segura com o conversor do TFLite e com a pipeline de correção automática.
  * **Compreensão das Técnicas de Quantização:**  Entender a diferença prática e os impactos entre a Dynamic Range Quantization (INT8) e a Float16 Quantization. Foi necessário pesquisar como o hardware de destino lida com esses formatos para documentar corretamente qual técnica usar em CPUs simples vs. GPUs Mobile.