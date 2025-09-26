import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Dict, Optional
import warnings
import gc
from config import (
    MODEL_NAME,
    MAX_NEW_TOKENS,
    MAX_CONTEXT_LENGTH,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    DO_SAMPLE,
    REPETITION_PENALTY,
    USE_CHAT_TEMPLATE,
    SYSTEM_MESSAGE,
    LOAD_IN_8BIT,
    LOAD_IN_4BIT
)

# Suprimir warnings innecesarios
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class HuggingFaceClient:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = MODEL_NAME
        print(f"Dispositivo detectado: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def _get_quantization_config(self):
        """Configurar cuantización para ahorrar memoria"""
        if not (LOAD_IN_8BIT or LOAD_IN_4BIT) or self.device == "cpu":
            return None
            
        try:
            if LOAD_IN_4BIT:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif LOAD_IN_8BIT:
                return BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            print("bitsandbytes no disponible, cargando en precisión completa")
            return None
    
    def load_model(self):
        """Cargar modelo y tokenizer optimizado para modelos instruction-tuned"""
        try:
            print(f"Cargando modelo robusto: {self.model_name}")
            print("Descargando modelo...")
            
            # Cargar tokenizer
            print("Cargando tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"  # Mejor para generación
            )
            
            # Configurar tokens especiales
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configuración de cuantización
            quantization_config = self._get_quantization_config()
            
            # Configurar parámetros del modelo
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "quantization_config": quantization_config,
            }
            
            # Configurar device_map
            if self.device == "cuda" and not (LOAD_IN_8BIT or LOAD_IN_4BIT):
                model_kwargs["device_map"] = "auto"
            
            print("Cargando modelo en memoria...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Mover a dispositivo si es necesario
            if self.device == "cpu" or (LOAD_IN_8BIT or LOAD_IN_4BIT):
                if not (LOAD_IN_8BIT or LOAD_IN_4BIT):
                    self.model = self.model.to(self.device)
            
            print("Modelo cargado exitosamente")
            self._print_model_info()
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
    
    def _print_model_info(self):
        """Mostrar información detallada del modelo"""
        if self.model is None:
            return
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"Información del modelo:")
            print(f"   • Nombre: {self.model_name}")
            print(f"   • Parámetros totales: {total_params:,}")
            print(f"   • Parámetros entrenables: {trainable_params:,}")
            print(f"   • Dispositivo: {self.device}")
            
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"   • Memoria GPU usada: {memory_allocated:.1f}GB")
                
        except Exception as e:
            print(f"No se pudo obtener info del modelo: {e}")
    
    def _prepare_messages(self, user_message: str, context_messages: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """Preparar mensajes en formato de chat estándar"""
        messages = []
        
        # Agregar mensaje de sistema
        messages.append({
            "role": "system", 
            "content": SYSTEM_MESSAGE
        })
        
        # Agregar contexto de conversaciones anteriores
        if context_messages:
            # Limitar contexto para no exceder límites
            recent_context = context_messages[-(MAX_CONTEXT_LENGTH // 100):]
            messages.extend(recent_context)
        
        # Agregar mensaje actual
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def generate_response(self, user_message: str, context_messages: List[Dict[str, str]] = None) -> str:
        """Generar respuesta usando el modelo robusto"""
        if self.model is None or self.tokenizer is None:
            return "Error: Modelo no está cargado. Reinicia la aplicación."
        
        try:
            # Preparar mensajes
            messages = self._prepare_messages(user_message, context_messages)
            
            # Aplicar plantilla de chat si el modelo lo soporta
            if USE_CHAT_TEMPLATE and hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    # Fallback a formato simple
                    prompt = self._format_messages_simple(messages)
            else:
                prompt = self._format_messages_simple(messages)
            
            # Tokenizar
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Verificar límites de contexto
            if inputs.shape[1] > MAX_CONTEXT_LENGTH:
                print("Contexto muy largo, truncando...")
                inputs = inputs[:, -MAX_CONTEXT_LENGTH:]
            
            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=DO_SAMPLE,
                    repetition_penalty=REPETITION_PENALTY,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decodificar solo la nueva parte
            new_tokens = outputs[0][inputs.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Limpiar respuesta
            response = self._clean_response(response)
            
            # Limpiar memoria GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return "Memoria GPU insuficiente. Intenta: 'clear', reiniciar, o cambiar a un modelo más pequeño."
        
        except Exception as e:
            return f"Error generando respuesta: {e}"
    
    def _format_messages_simple(self, messages: List[Dict[str, str]]) -> str:
        """Formato simple de mensajes cuando no hay plantilla de chat"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"Sistema: {content}\n\n"
            elif role == "user":
                formatted += f"Usuario: {content}\n"
            elif role == "assistant":
                formatted += f"Asistente: {content}\n"
        
        formatted += "Asistente:"
        return formatted
    
    def _clean_response(self, response: str) -> str:
        """Limpiar y optimizar la respuesta"""
        # Eliminar tokens especiales residuales
        response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        
        # Eliminar prefijos no deseados
        if response.startswith("Asistente:"):
            response = response[10:].strip()
        
        # Eliminar repeticiones obvias al final
        lines = response.split('\n')
        if len(lines) > 1 and lines[-1] == lines[-2]:
            lines = lines[:-1]
        
        response = '\n'.join(lines).strip()
        
        # Truncar si es muy largo
        if len(response) > 1000:
            response = response[:1000] + "..."
        
        # Validar que no esté vacío
        if not response.strip():
            response = "Lo siento, no pude generar una respuesta apropiada. ¿Podrías reformular tu pregunta?"
        
        return response
    
    def get_model_info(self) -> str:
        """Información detallada del modelo"""
        if self.model is None:
            return "Ningún modelo cargado"
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            memory_info = ""
            
            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_info = f" | GPU: {memory_used:.1f}GB"
            
            quantization = ""
            if LOAD_IN_4BIT:
                quantization = " | 4-bit"
            elif LOAD_IN_8BIT:
                quantization = " | 8-bit"
            
            return f"{self.model_name} | {total_params:,} parámetros | {self.device}{memory_info}{quantization}"
            
        except Exception:
            return f"{self.model_name} | {self.device}"
    
    def clear_memory(self):
        """Limpiar memoria GPU/CPU"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print("Memoria limpiada")