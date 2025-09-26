import sys
import signal
from memory_system import MemorySystem
from huggingface_client import HuggingFaceClient


class ConversationalAgent:
    def __init__(self):
        self.memory = MemorySystem()
        self.hf_client = HuggingFaceClient()
        self.running = True
        
        # Configurar manejo de señales para salida elegante
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Manejar Ctrl+C para salida elegante"""
        print("\n\n¡Hasta luego!")
        sys.exit(0)
    
    def show_welcome_message(self):
        """Mostrar mensaje de bienvenida"""
        print("=" * 70)
        print("AGENTE CONVERSACIONAL CON MEMORIA (Hugging Face)")
        print("=" * 70)
        print("¡Hola! Soy tu asistente conversacional.")
        print("\nComandos especiales:")
        print("  • 'exit' o 'quit' - Salir de la aplicación")
        print("  • 'clear' - Limpiar memoria de corto plazo")
        print("  • 'history' - Mostrar historial reciente")
        print("  • 'info' - Mostrar información del modelo")
        print("\nEscribe tu mensaje y presiona Enter...")
        print("=" * 70)
    
    def show_thinking(self):
        """Mostrar indicador de procesamiento"""
        print("Generando respuesta...", end="", flush=True)
    
    def clear_thinking(self):
        """Limpiar indicador de procesamiento"""
        print("\r" + " " * 25 + "\r", end="", flush=True)
    
    def process_special_command(self, user_input: str):
        """Procesar comandos especiales. Retorna True para exit, 'COMMAND_PROCESSED' para otros comandos, False para continuar."""
        command = user_input.lower().strip()
        
        if command in ['exit', 'quit']:
            print("\n¡Hasta luego!")
            return True
        
        elif command == 'clear':
            self.memory.clear_short_term()
            return "COMMAND_PROCESSED"  # Comando procesado, no generar respuesta
        
        elif command == 'history':
            self.memory.show_recent_history()
            return "COMMAND_PROCESSED"  # Comando procesado, no generar respuesta
        
        elif command == 'info':
            info = self.hf_client.get_model_info()
            print(f"\n{info}\n")
            return "COMMAND_PROCESSED"  # Comando procesado, no generar respuesta
        
        return False  # No es un comando especial, continuar con generación
    
    def initialize(self):
        """Inicializar el agente"""
        # Mostrar mensaje de bienvenida
        self.show_welcome_message()
        
        # Cargar modelo de Hugging Face
        print("Cargando modelo de lenguaje...")
        
        if not self.hf_client.load_model():
            print("No se pudo cargar el modelo. Verifica tu conexión a internet.")
            return False
        
        # Mostrar información del modelo
        model_info = self.hf_client.get_model_info()
        print(f"{model_info}")
        
        # Cargar memoria de conversaciones anteriores
        print("Cargando memoria de conversaciones anteriores...")
        self.memory.load_from_long_term()
        
        print("\nValidaciones completas")
        print("-" * 70)
        return True
    
    def run(self):
        """Ejecutar el bucle principal de conversación"""
        if not self.initialize():
            return
        
        while self.running:
            try:
                # Obtener entrada del usuario
                user_input = input("\nTú: ").strip()
                
                if not user_input:
                    continue
                
                # Procesar comandos especiales
                command_result = self.process_special_command(user_input)
                if command_result == True:  # exit/quit
                    break
                elif command_result == "COMMAND_PROCESSED":  # comando ejecutado, no generar respuesta
                    continue
                
                # Mostrar indicador de procesamiento
                self.show_thinking()
                
                # Obtener contexto de memoria
                context_messages = self.memory.get_context_for_prompt()
                
                # Generar respuesta
                response = self.hf_client.generate_response(user_input, context_messages)
                
                # Limpiar indicador de procesamiento
                self.clear_thinking()
                
                # Mostrar respuesta
                print(f"Asistente: {response}")
                
                # Guardar interacción en memoria solo si la respuesta es válida
                if response and not response.startswith("Error"):
                    self.memory.add_interaction(user_input, response)
                
            except KeyboardInterrupt:
                # Ya manejado por signal_handler
                break
            except EOFError:
                print("\n\n¡Hasta luego!")
                break
            except Exception as e:
                self.clear_thinking()
                print(f"\n❌ Error inesperado: {e}")
                print("Intenta de nuevo...")


def main():
    """Función principal"""
    print("🚀 Iniciando Agente Conversacional con Hugging Face...")
    
    # Verificar dependencias
    try:
        import torch
        import transformers
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"Error: Falta dependencia - {e}")
        print("Ejecuta: pip install -r requirements.txt")
        return
    
    agent = ConversationalAgent()
    agent.run()


if __name__ == "__main__":
    main()