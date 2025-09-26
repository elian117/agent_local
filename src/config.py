# =====================================================
# MODELOS
# ----------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_NAME = "google/gemma-2b-it"

# =====================================================
# CONFIGURACIÓN
# =====================================================

# Configuración de memoria
SHORT_TERM_MEMORY_LIMIT = 10  # Últimas 10 interacciones
MEMORY_FILE_PATH = "data/conversation_history.json"

# Configuración del modelo
MAX_NEW_TOKENS = 256         # Tokens máximos para la respuesta
MAX_CONTEXT_LENGTH = 2048    # Contexto máximo
TEMPERATURE = 0.8            # Creatividad (0.1 = conservador, 0.9 = creativo)
TOP_P = 0.9                  # Núcleo de probabilidad
TOP_K = 50                   # Top-k sampling
DO_SAMPLE = True             # Activar sampling
REPETITION_PENALTY = 1.05    # Penalizar repeticiones

# Configuración de sistema
USE_CHAT_TEMPLATE = True     # Usar plantilla de chat del modelo
SYSTEM_MESSAGE = """Eres un asistente conversacional útil, honesto y amigable. Respondes de manera clara y precisa."""

# Configuración de hardware
LOAD_IN_8BIT = False         # Cargar en 8-bit para ahorrar memoria
LOAD_IN_4BIT = False         # Cargar en 4-bit para máximo ahorro de memoria
