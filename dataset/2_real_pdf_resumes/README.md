# 2. real_pdf_resumes/

## Dataset
Resume Classification Dataset for NLP

## Finalidad
Entrenar el sistema para trabajar con PDFs reales y formatos variados.

## Qué aprenderá la red neuronal

### Extracción de texto de PDFs
- Manejar diferentes estructuras de PDF (texto seleccionable vs escaneado)
- Preservar formato y jerarquía durante la extracción
- Manejar caracteres especiales, acentos y símbolos

### Adaptación a formatos variados
- Reconocer diferentes plantillas de CV (modernas, tradicionales, creativas)
- Identificar secciones aunque estén etiquetadas diferente (Experiencia/Experiencia Laboral/Historial Laboral)
- Manejar formatos diversos: tablas, listas, párrafos, encabezados

### Robustez y tolerancia a errores
- Procesar CVs con estructuras no estándar
- Manejar texto mal formateado o con caracteres corruptos
- Ignorar ruido visual (logos, marcas de agua, elementos decorativos)

### Limpieza de texto
- Eliminar información irrelevante (headers/footers de páginas)
- Normalizar espacios y saltos de línea
- Unificar formatos de fecha y números

## Uso dentro del proyecto
```text
PDF → extracción de texto → limpieza NLP
```

## Link de descarga
https://www.kaggle.com/datasets/hassnainzaidi/resume-classification-dataset-for-nlp