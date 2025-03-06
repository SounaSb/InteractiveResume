import requests
import json
import logging
from typing import List, Dict, Generator
from pathlib import Path
from .retrieval import RagRetriever
import os
from openai import OpenAI



# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('rag.generation')

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"



# Read base system prompt
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
with open(SYSTEM_PROMPT_PATH, 'r') as f:
    BASE_SYSTEM_PROMPT = f.read().strip()



retriever = RagRetriever(media_folder=str(Path(__file__).parent.parent / "ragdb"))
logger.debug(f"Media folder: {str(Path(__file__).parent.parent / 'ragdb')}")  






def get_streaming_response(message: str, history: List[Dict[str, str]]) -> Generator[str, None, None]:
    try:
        logger.debug(f"Received message: {message}")
        context = retriever.get_top_chunks(message, k=5)
        
        # Check if context is empty or irrelevant
        if not context.strip():
            yield "I don't have specific information about that in the context."
            return
        
        # Format chat history (last 2 exchanges)
        chat_history = ""
        if history:
            recent_history = history[-2:]  # Get last 2 exchanges
            for entry in recent_history:
                chat_history += f"User: {entry['text']}\n" if entry['sender'] == 'user' else f"Assistant: {entry['text']}\n"
        
        
        # Include chat history in the prompt
        prompt = f"Previous conversation:\n{chat_history}\n\nAnswer following question: {message}\n\nUsing information from this context: \n{context}"
        
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "system": BASE_SYSTEM_PROMPT,
            "stream": True,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.65,   # Reduce for more focused responses
                "top_p": 0.7,          # Reduce for more deterministic outputs
                "max_tokens": 300,     # Reduce to encourage conciseness
                "stop": ["###", "\n\n\n"]
            }
        }

        response = requests.post(OLLAMA_ENDPOINT, json=payload, stream=True)

        if response.status_code != 200:
            logger.error(f"Error: {response.status_code} - {response.text}")
            yield "Sorry, I encountered an error."

        else:
            for line in response.iter_lines():
                if line:
                    try:
                        response_data = json.loads(line.decode('utf-8'))
                        if 'response' in response_data:
                            yield response_data['response']
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error: {e}")
                        continue
                    

    except Exception as e:
        logger.error(f"Error in get_streaming_response: {str(e)}")
        yield "Sorry, I encountered an unexpected error."






model_gpt = "gpt-o3-mini"
# model_gpt = "gpt-o3-turbo"

def get_streaming_response_api(message: str, history: List[Dict[str, str]], api_key: str, model: str = "gpt-o3-mini") -> Generator[str, None, None]:
    try:
        logger.debug(f"Received message for OpenAI API: {message}")
        

        # Check if this is just a greeting first
        simple_greetings = ["hi", "hello", "hey", "yo", "sup", "what's up", "howdy"]
        if message.lower().strip() in simple_greetings:
            logger.debug("Simple greeting detected")
            yield "Hello! How can I help you learn about Daniel's background today?"
            return
        
        
        context = retriever.get_top_chunks(message, k=4)
        
        # Check if context is empty or irrelevant
        if not context.strip():
            yield "I don't have specific information about that in the context."
            return
        
        # Format chat history (last 2 exchanges)
        messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT}
        ]
        
        # Format chat history (last 2 exchanges)
        chat_history = ""
        if history:
            recent_history = history[-2:]  # Get last 2 exchanges
            for entry in recent_history:
                chat_history += f"User: {entry['text']}\n" if entry['sender'] == 'user' else f"Assistant: {entry['text']}\n"
        
        
        # Include context and question
        messages.append({
            "role": "user", 
            "content": f"Previous Conversation: {chat_history}\n\nAnswer the following question: {message}\n\nUsing information you deem directly relevant regarding the question from this context: \n{context}"
        })
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.75,
            max_tokens=350,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
    except Exception as e:
        logger.error(f"Error in get_streaming_response_api: {str(e)}")
        yield "Sorry, I encountered an unexpected error with the OpenAI API."
