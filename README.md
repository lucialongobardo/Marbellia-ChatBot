
![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Final Project | Marbella Turism Q&A Chatbot

## Project Overview

This project builds a YouTube-based Q&A chatbot using Gradio, LangChain, Pinecone, and OpenAI. The chatbot provides answers about turism in Marbella city using internal document vectors from relevant YouTube videos' transcriptions about Marbella city. For general queries agent uses web search for responses through SerpApi. Users can update the knowledgebase by adding new url youtube videos to the urls.txt file.

Please visit the following link to make use of this agent: https://huggingface.co/spaces/longobardomartin/proyectofinal (if not working please proceed to Usage description here)

## Table of Contents

   - Folder Structure
   - Environment Setup
   - Project Architecture   
   - Usage

## Folder Structure

- app.py - Main script to run the chatbot.
- knowledgebase.py - Script to get transcripts from YouTube videos and storing the data in Pinecone.
- agent.py - Script to specify the agent behaivour.
- utils.py - Script for auxiliary functions.
- urls.txt - Text file containing YouTube video links used in the project.
- requirements.txt - Fille containing libraries and dependencies to be installed locally.
- README.md - Project documentation (this file).
- Marbella turism.pdf - Presentation slides for the Final Project.
- .env - File containing global enviroment variables like api keys.

## Environment Setup

Add your environment variables by setting up a .env file with:
   - OPENAI_API_KEY: API key for OpenAI.
   - LANGCHAIN_API_KEY: API key for LangChain.
   - PINECONE_API_KEY: API key for Pinecone.
   - SERPAPI_API_KEY: API key for SerpAPI.

## Project Architecture

The chatbot uses the following architecture:

- Data Retrieval: Combines a vector database (Pinecone) for structured data retrieval and SerpAPI for web search.

- Routing: Uses LangChain's agent to dynamically route user questions to the appropriate source (either vector store or web search) using Tools and Prompts templates.

- Memory: ConversationBufferMemory maintains chat history of recent exchanged messages for context.

- LLM Integration: GPT-4 processes user queries and generates responses. Huggingface transformer summarizes search results.

## Usage

- Use requirements.txt to install the necessary dependencies.

- Add video links to urls.txt (one link per line).

- Run the knowledgebase.py script to generate transcriptions, create embeddings, and set up the chatbot interface.

- Run app.py script to render the app locally.

- Interact with the chatbot through Gradio accessing at the provided localhost url in your browser.

## Presentation

- Marbellia Final Project.pdf

