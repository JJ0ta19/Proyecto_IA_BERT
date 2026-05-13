# 3. job_resume_matching/

## Dataset
Job Resume Fit Dataset

## Finalidad
Entrenar el sistema de recomendación laboral y matching semántico.

## Qué aprenderá la red neuronal

### Matching semántico
- Calcular similaridad entre el contenido del CV y la descripción del puesto
- Comprender que "desarrollador web" y "programador frontend" son relacionados
- Manejar sinónimos y variaciones de términos técnicos

### Análisis de compatibilidad
- Identificar gaps de skills entre lo que ofrece el candidato y lo que requiere el puesto
- Ponderar la importancia relativa de diferentes requirements
- Generar score de compatibilidad (0-100%)

### Relación skills - ofertas
- Mapear skills técnicos a requisitos de trabajo específicos
- Identificar skills equivalents (Python = R para data science)
- Detectar skills opcionales vs obligatorios en ofertas

### Recomendación inteligente
- Rankear múltiples ofertas laborales para un candidato
- Sugerir skills faltantes que el candidato podría desarrollar
- Identificar trayectorias profesionales relacionadas

### Embeddings profesionales
- Crear representaciones vectoriales de perfiles profesionales
- Medir distancia semántica entre campos laborales relacionados
- Agrupar profesiones y skills relacionados

## Uso dentro del proyecto
```text
Resume + Job Description → porcentaje de compatibilidad
```

## Link de descarga
https://huggingface.co/datasets/batuhanmtl/job_resume_fit