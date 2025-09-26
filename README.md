# Agente Conversacional CLI con Memoria (Hugging Face)

Un agente conversacional de línea de comandos que utiliza modelos de lenguaje abiertos de Hugging Face e implementa un sistema de memoria persistente para mantener contexto entre conversaciones.

## 🚀 Características

- **Memoria de Corto Plazo**: Mantiene las últimas 10 interacciones en memoria RAM para contexto inmediato
- **Memoria de Largo Plazo**: Persiste todo el historial de conversación en archivo JSON local
- **Modelos Abiertos**: Utiliza modelos de Hugging Face (DialoGPT, BlenderBot, etc.)
- **Ejecución Local**: Funciona completamente offline después de la descarga inicial
- **Interfaz CLI Interactiva**: Interfaz de terminal amigable con comandos especiales
- **Optimización Automática**: Detección de GPU/CPU y manejo inteligente de memoria
- **Recuperación de Sesión**: El agente recuerda conversaciones anteriores al reiniciar

## 📋 Requisitos del Sistema

### Requisitos Mínimos
- Python 3.8+
- 4GB RAM
- 2GB espacio libre (para descargar modelos)
- Conexión a internet (solo para descarga inicial)

### Requisitos Recomendados
- Python 3.9+
- 8GB+ RAM
- GPU con CUDA (opcional, mejora velocidad significativamente)
- 5GB espacio libre

## 🛠️ Instalación

### 1. Clonar o descargar el proyecto
```bash
git clone <url-del-repositorio>
cd agente-conversacional-hf
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

**Para usuarios con GPU NVIDIA (recomendado):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Primera ejecución
```bash
python main.py
```

**Nota**: La primera vez descargará el modelo (~1GB-3GB) automáticamente.

## 🎯 Uso

### Ejecutar el agente
```bash
python main.py
```

### Comandos disponibles

Durante la conversación, puedes usar estos comandos especiales:

- `exit` o `quit` - Terminar la aplicación
- `clear` - Limpiar memoria de corto plazo
- `history` - Mostrar las últimas 5 interacciones
- `info` - Mostrar información del modelo cargado

```

## 🏗️ Arquitectura del Sistema

### Estructura del proyecto
```
proyecto/
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── config.py              # Configuración del modelo HF y parámetros
│   ├── memory_system.py       # Sistema de memoria (sin cambios)
│   ├── huggingface_client.py  # Cliente para modelos de Hugging Face
│   └── cli_interface.py       # Interfaz CLI adaptada para HF
└── data/
    └── conversation_history.json  # Historial persistente
```

### Componentes principales

#### 1. Sistema de Memoria (`memory_system.py`)
- **Sin cambios** respecto a la versión Azure OpenAI
- Memoria de corto y largo plazo funcional

#### 2. Cliente Hugging Face (`huggingface_client.py`)
- **Carga automática de modelos**: Descarga y cache local
- **Detección de hardware**: GPU/CPU automático
- **Generación optimizada**: Control de temperatura, top-p, etc.
- **Manejo de contexto**: Truncado inteligente para evitar límites
- **Limpieza de respuestas**: Filtrado de repeticiones y formato

#### 3. Interfaz CLI Mejorada (`cli_interface.py`)
- **Verificación de dependencias**: Comprueba PyTorch/Transformers
- **Información del modelo**: Comando `info` para detalles técnicos
- **Mejor feedback**: Indicadores de descarga y carga

#### 4. Configuración (`config.py`)
- **Modelos configurables**: Fácil cambio entre modelos
- **Parámetros ajustables**: Temperatura, longitud, etc.
- **Límites inteligentes**: Manejo de memoria y contexto

## 🤖 Modelos Disponibles

### Modelos Recomendados (configurables en `config.py`):

```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "google/gemma-2b-it"
```

## 🔧 Decisiones Técnicas

### Selección de Modelo
- **Qwen2.5-1.5B-Instruct**: Balance entre calidad y recursos

### Optimizaciones de Rendimiento
- **Truncado inteligente**: Mantiene contexto relevante
- **Cache local**: Los modelos se descargan una sola vez
- **Generación controlada**: Evita respuestas muy largas

### Manejo de Memoria
- **Detección automática**: GPU vs CPU
- **Límites dinámicos**: Ajustados según el dispositivo
- **Limpieza automática**: Liberación de memoria entre respuestas

## 📊 Rendimiento Esperado

### En CPU (sistema típico):
- **Carga inicial**: 1-3 minutos
- **Respuesta**: 30-120 segundos
- **Memoria RAM**: 6-8 GB

## 🐛 Troubleshooting

### Problemas comunes

#### "Error cargando modelo"
```bash
# Solución 1: Actualizar transformers
pip install --upgrade transformers torch

# Solución 2: Limpiar cache
rm -rf ~/.cache/huggingface/transformers/
```

#### "CUDA out of memory"
- Cambiar a modelo más pequeño en `config.py`
- Usar CPU: `CUDA_VISIBLE_DEVICES="" python main.py`
- Reducir `MAX_LENGTH` en config

#### "Respuestas de baja calidad"
- Probar diferentes modelos en `config.py`
- Ajustar `TEMPERATURE` (0.7-0.9)
- Usar comando `clear` para limpiar contexto

#### "Demora en primera carga"
- Normal: descarga modelo desde internet
- Verificar espacio en disco (~2-5GB)
- Paciencia en la primera ejecución

## 📝 Configuración Personalizada

Para cambiar modelos o parámetros, edita `src/config.py`:

```python
# Cambiar modelo
MODEL_NAME = "modelo-preferido"

# Ajustar parámetros de generación
TEMPERATURE = 0.8      # Más creativo
MAX_LENGTH = 200       # Respuestas más largas
TOP_P = 0.95          # Más diversidad
```

---
