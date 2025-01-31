import gradio as gr
from agent import call_agent
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')


# Función del bot que procesa el mensaje del usuario
def chatbot(message, history=[]):
    # Agregar el mensaje del usuario al historial
    history.append(("Usuario:", message))
    # Consultar al agente de OpenAI
    response = call_agent(message)
    # Generar una respuesta simple del bot
    response = f"Bot:'{response}'"
    history.append((response,))
    # Formatear el historial como un bloque de texto
    chat_history = "\n".join([f"{msg[0]} {msg[1]}" if len(msg) > 1 else msg[0] for msg in history])
    return chat_history, history

# Interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Chatbot sencillo con Gradio")
    
    # Caja para mostrar el historial de mensajes
    chatbox = gr.Textbox(lines=10, label="Historial de mensajes", interactive=False)
    
    # Caja para escribir mensajes
    input_box = gr.Textbox(lines=1, placeholder="Escribe tu mensaje aquí", label="Mensaje")
        
    # Almacenamiento interno para el historial de chat
    state = gr.State([])

    # Lógica al presionar Enter en la caja de texto
    input_box.submit(chatbot, inputs=[input_box, state], outputs=[chatbox, state])

# Ejecutar la aplicación
demo.launch()