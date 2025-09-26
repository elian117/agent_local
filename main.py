"""
Agente Conversacional CLI con Memoria
Prueba Técnica - Backend Engineer (Versión Hugging Face)

Este script ejecuta el agente conversacional con memoria persistente
que utiliza modelos de lenguaje abiertos de Hugging Face.
"""

import sys
import os

# Agregar src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cli_interface import main

if __name__ == "__main__":
    main()