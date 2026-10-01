# Implementación del modelo BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.

**Estudiantes:** Juan Camilo Perez, Carlos Eduardo Trujillo, Osvaldo Marin

## Resumen

Este proyecto implementa el modelo **BART (Bidirectional and Auto-Regressive Transformer)** para la generación de **resúmenes abstractivos de noticias y textos en inglés**. Se utiliza el modelo preentrenado `facebook/bart-large-cnn`, integrado mediante **Hugging Face Transformers**, y se configura la generación mediante **Beam Search**.

El sistema fue evaluado utilizando **ROUGE-1, ROUGE-2 y ROUGE-L**, junto con métricas complementarias **BERTScore** y **METEOR**. Finalmente, se desarrolló una interfaz interactiva con **Streamlit** para permitir la generación de resúmenes sin requerir conocimientos técnicos especializados.

## Introducción

El proyecto toma como base el artículo **“BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension”**, propuesto por Lewis *et al.*. BART combina un **encoder bidireccional** con un **decoder autorregresivo**, lo que permite abordar tareas de comprensión y generación de lenguaje natural como el resumen abstractivo.

- **Artículo base:** https://aclanthology.org/2020.acl-main.703/
- **Repositorio original de BART en Fairseq:** https://github.com/facebookresearch/fairseq/tree/main/examples/bart
- **Modelo utilizado:** https://huggingface.co/facebook/bart-large-cnn

La problemática abordada se relaciona con el crecimiento constante de la información disponible en medios digitales y la necesidad de identificar rápidamente las ideas principales de textos extensos. Por esta razón, el objetivo del proyecto es **implementar y evaluar un sistema de generación automática de resúmenes abstractivos de noticias y textos en inglés mediante BART, integrando métricas de evaluación y una interfaz interactiva en Streamlit**.

## Marco teórico

### Arquitectura Transformer y mecanismo de atención

La arquitectura **Transformer**, propuesta por Vaswani *et al.* (2017), se fundamenta principalmente en el mecanismo de **Self-Attention**, que permite establecer relaciones entre los tokens de una secuencia independientemente de su distancia.

El mecanismo de atención utiliza tres representaciones:

- **Query (Q):** representa el elemento que busca información.
- **Key (K):** representa los elementos candidatos a ser atendidos.
- **Value (V):** contiene la información que finalmente será ponderada.

La atención se calcula mediante:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

### Arquitectura Encoder-Decoder de BART

BART utiliza una arquitectura **sequence-to-sequence** compuesta por:

- **Encoder bidireccional:** procesa el contexto completo mediante Self-Attention.
- **Decoder autorregresivo:** genera la salida token por token mediante Masked Self-Attention.
- **Cross-Attention:** permite que el decoder consulte las representaciones generadas por el encoder.

En **BART Large**, arquitectura base de `facebook/bart-large-cnn`, se utilizan **12 capas en el encoder y 12 capas en el decoder**, con una representación oculta de 1024 dimensiones.

<p align="center">
  <img src="assets/transformer_encoder_decoder.png" width="430" alt="Arquitectura Transformer Encoder-Decoder">
</p>

**Figura 1.** Arquitectura Transformer Encoder-Decoder. Fuente: [8].

### Preentrenamiento e innovaciones de BART

BART se preentrena como un **denoising autoencoder**, alterando deliberadamente el texto de entrada para posteriormente aprender a reconstruirlo. Entre las estrategias utilizadas se encuentran:

- Token Masking.
- Token Deletion.
- Text Infilling.
- Sentence Permutation.
- Document Rotation.

Entre estas estrategias, **Text Infilling** y la permutación de oraciones mostraron un comportamiento especialmente consistente para diferentes tareas de procesamiento de lenguaje natural.

## Metodología

### Implementación

El flujo de trabajo implementado fue:

1. Carga del modelo y tokenizer preentrenados.
2. Ingreso del texto en inglés.
3. Tokenización.
4. Generación del resumen mediante Beam Search.
5. Decodificación de la salida.
6. Evaluación mediante métricas.
7. Visualización del resultado mediante Streamlit.

### Herramientas utilizadas

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Streamlit**
- **sentencepiece**
- **safetensors**
- **NLTK**
- **BERTScore**

### Uso de pesos preentrenados

El proyecto utiliza `facebook/bart-large-cnn`, una variante de BART Large ajustada para resumen automático sobre **CNN/DailyMail**. Los pesos y el tokenizer se cargan directamente desde Hugging Face mediante:

```python
BartTokenizer.from_pretrained("facebook/bart-large-cnn")
BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
```

El uso de pesos preentrenados permite aplicar **transfer learning** y evita el costo computacional asociado con entrenar el modelo completo desde cero.

### Preprocesamiento

El tokenizer convierte el texto de entrada en:

- `input_ids`: identificadores numéricos de los tokens.
- `attention_mask`: máscara que indica qué posiciones contienen información válida.

La entrada se limita a un máximo de **1024 tokens**, aplicando truncamiento cuando sea necesario.

### Generación del resumen

La inferencia se realiza con `generate()` y Beam Search utilizando los principales parámetros:

```python
num_beams=6
length_penalty=2.0
no_repeat_ngram_size=3
early_stopping=True
```

La longitud mínima y máxima se calcula dinámicamente en función de la cantidad aproximada de palabras solicitada por el usuario.

### Evaluación

La evaluación principal utiliza:

- **ROUGE-1:** coincidencia de unigramas.
- **ROUGE-2:** coincidencia de bigramas.
- **ROUGE-L:** subsecuencia común más larga.

Como métricas complementarias se incorporaron:

- **BERTScore:** similitud semántica entre el resumen generado y la referencia humana.
- **METEOR:** correspondencia léxica y variaciones morfológicas.

## Desarrollo e implementación

### Requisitos

Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

### Ejecución

Ejecutar la aplicación con:

```bash
streamlit run Bart_Implementacion_resumenes.py
```

Después:

1. Abrir la dirección local indicada por Streamlit.
2. Ingresar un texto en inglés.
3. Seleccionar la longitud aproximada del resumen.
4. Presionar **Generar resumen**.

Durante la primera ejecución se descargan automáticamente los pesos y el tokenizer de `facebook/bart-large-cnn`.

## Resultados y análisis

Para evaluar el sistema se utilizó un artículo periodístico en inglés y se comparó el resumen generado por BART con un resumen humano de referencia.

### Resultados ROUGE

<p align="center">
  <img src="assets/rouge_results.png" width="620" alt="Resultados ROUGE">
</p>

**Figura 2.** Resultados obtenidos con ROUGE.

Los F1-Score obtenidos fueron:

| Métrica | F1-Score |
|---|---:|
| ROUGE-1 | 0.4416 |
| ROUGE-2 | 0.2933 |
| ROUGE-L | 0.3117 |

En las tres métricas, la **precisión fue superior al recall**, indicando que una proporción importante del contenido seleccionado por el modelo coincide con la referencia humana, aunque algunos elementos de la referencia no son recuperados en el resumen generado.

### Métricas complementarias

<p align="center">
  <img src="assets/complementary_metrics.png" width="720" alt="BERTScore y METEOR">
</p>

**Figura 3.** Resultados de BERTScore y METEOR.

| Métrica | Resultado |
|---|---:|
| BERTScore F1 | 0.8932 |
| METEOR | 0.3515 |

El **BERTScore elevado** indica una alta similitud semántica con el resumen humano, mientras que el valor de **METEOR** refleja una coincidencia léxica más moderada. En conjunto, los resultados sugieren que el modelo conserva mejor el **significado general** que la redacción literal, comportamiento coherente con una tarea de resumen abstractivo.

### Interfaz desarrollada

<p align="center">
  <img src="assets/streamlit_interface.png" width="850" alt="Interfaz Streamlit del generador de resúmenes">
</p>

**Figura 4.** Interfaz interactiva desarrollada en Streamlit.

## Conclusiones

La implementación permitió comprobar que **BART es adecuado para tareas de resumen abstractivo**, gracias a la combinación de un encoder bidireccional, un decoder autorregresivo y mecanismos de atención.

El uso de **pesos preentrenados y transfer learning** permitió construir un sistema funcional sin entrenar el modelo desde cero. Las métricas **ROUGE, BERTScore y METEOR** permitieron analizar el desempeño desde perspectivas léxicas y semánticas.

Entre las principales limitaciones se encuentran el número reducido de pruebas, la imposibilidad de garantizar factualidad únicamente mediante las métricas implementadas y el hecho de que `facebook/bart-large-cnn` fue entrenado y ajustado principalmente sobre textos en **inglés**, por lo que su desempeño no se garantiza para otros idiomas sin una adaptación y evaluación específica.

Como trabajo futuro se propone ampliar el conjunto de evaluación, incorporar métricas de **consistencia factual**, comparar BART con otros modelos y analizar diferentes configuraciones de los parámetros de generación.

Finalmente, la interfaz desarrollada en **Streamlit** demuestra la viabilidad de integrar BART en una aplicación práctica y accesible para usuarios no técnicos.

## Estructura recomendada del repositorio

```text
proyecto-bart/
│
├── README.md
├── Bart_Implementacion_resumenes.py
├── requirements.txt
├── Perez_Trujillo_Marin_Implementacion_BART.ipynb
│
└── assets/
    ├── transformer_encoder_decoder.png
    ├── rouge_results.png
    ├── complementary_metrics.png
    └── streamlit_interface.png
```

## Referencias

[1] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 7871–7880, doi: 10.18653/v1/2020.acl-main.703.

[2] A. Vaswani *et al.*, “Attention is all you need,” in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017, pp. 5998–6008.

[3] Meta AI, “facebook/bart-large-cnn,” *Hugging Face*. [Online]. Available: https://huggingface.co/facebook/bart-large-cnn. [Accessed: Oct. 1, 2026].

[4] C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in *Text Summarization Branches Out: Proceedings of the ACL-04 Workshop*, Barcelona, Spain, 2004, pp. 74–81.

[5] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “BERTScore: Evaluating text generation with BERT,” in *Proc. International Conference on Learning Representations (ICLR)*, 2020.

[6] S. Banerjee and A. Lavie, “METEOR: An automatic metric for MT evaluation with improved correlation with human judgments,” in *Proc. ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*, Ann Arbor, MI, USA, 2005, pp. 65–72.

[7] T. Wolf *et al.*, “Transformers: State-of-the-art natural language processing,” in *Proc. 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP)*, 2020, pp. 38–45.

[8] N. Velandia, “Multihead_Transformer_2026_2_Ma,” Jupyter Notebook, curso *Procesamiento de Datos Secuenciales con Deep Learning*, Universidad Autónoma de Occidente, 2026.

[9] Streamlit, “Streamlit documentation.” [Online]. Available: https://docs.streamlit.io/. [Accessed: Oct. 1, 2026].
