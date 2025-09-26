import json
import os
from datetime import datetime
from typing import List, Dict
from config import SHORT_TERM_MEMORY_LIMIT, MEMORY_FILE_PATH


class MemorySystem:
    def __init__(self):
        self.short_term_memory: List[Dict] = []
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Crear directorio de datos si no existe"""
        os.makedirs(os.path.dirname(MEMORY_FILE_PATH), exist_ok=True)
    
    def add_interaction(self, user_message: str, assistant_response: str):
        """Agregar nueva interacción a la memoria"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response
        }
        
        # Agregar a memoria de corto plazo
        self.short_term_memory.append(interaction)
        
        # Mantener límite de memoria corto plazo
        if len(self.short_term_memory) > SHORT_TERM_MEMORY_LIMIT:
            self.short_term_memory.pop(0)
        
        # Persistir en memoria de largo plazo
        self._save_to_long_term(interaction)
    
    def _save_to_long_term(self, interaction: Dict):
        """Guardar interacción en archivo persistente"""
        try:
            # Cargar historial existente o crear nuevo
            if os.path.exists(MEMORY_FILE_PATH):
                with open(MEMORY_FILE_PATH, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Agregar nueva interacción
            history.append(interaction)
            
            # Guardar historial actualizado
            with open(MEMORY_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error al guardar en memoria de largo plazo: {e}")
    
    def load_from_long_term(self):
        """Cargar historial desde archivo persistente al inicio"""
        try:
            if os.path.exists(MEMORY_FILE_PATH):
                with open(MEMORY_FILE_PATH, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    
                # Cargar últimas interacciones a memoria de corto plazo
                recent_interactions = history[-SHORT_TERM_MEMORY_LIMIT:]
                self.short_term_memory = recent_interactions
                
                print(f"Memoria cargada: {len(history)} conversaciones en total, {len(recent_interactions)} en memoria activa")
            else:
                print("No se encontró historial previo. Iniciando nueva sesión.")
        except Exception as e:
            print(f"Error al cargar memoria de largo plazo: {e}")
    
    def get_context_for_prompt(self) -> List[Dict[str, str]]:
        """Obtener contexto de memoria para incluir en el prompt"""
        context_messages = []
        
        for interaction in self.short_term_memory:
            context_messages.append({
                "role": "user",
                "content": interaction["user_message"]
            })
            context_messages.append({
                "role": "assistant", 
                "content": interaction["assistant_response"]
            })
        
        return context_messages
    
    def clear_short_term(self):
        """Limpiar memoria de corto plazo"""
        self.short_term_memory = []
        print("Memoria de corto plazo limpiada.")
    
    def show_recent_history(self, limit: int = 5):
        """Mostrar historial reciente"""
        if not self.short_term_memory:
            print("No hay historial reciente.")
            return
        
        print(f"\n--- Historial reciente (últimas {min(limit, len(self.short_term_memory))} interacciones) ---")
        recent = self.short_term_memory[-limit:]
        
        for i, interaction in enumerate(recent, 1):
            timestamp = interaction["timestamp"]
            user_msg = interaction["user_message"][:100] + "..." if len(interaction["user_message"]) > 100 else interaction["user_message"]
            assistant_msg = interaction["assistant_response"][:100] + "..." if len(interaction["assistant_response"]) > 100 else interaction["assistant_response"]
            
            print(f"{i}. [{timestamp}]")
            print(f"   Usuario: {user_msg}")
            print(f"   Asistente: {assistant_msg}")
            print()
        print("--- Fin del historial ---\n")
