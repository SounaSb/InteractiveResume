import logging
import os
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from rag.generation import get_streaming_response_api

def home(request):
    return render(request, 'home.html')

def chat(request):
    return render(request, 'chat.html')

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        # Get OpenAI API key from environment variables
        api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            return JsonResponse({"error": "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."}, 
                              status=500)
        
        data = json.loads(request.body)
        message = data.get('message', '')
        history = data.get('history', [])
        model = data.get('model', 'gpt-3.5-turbo')  # Default to GPT-3.5 if not specified
        
        return StreamingHttpResponse(
            # get_streaming_response(message, history),
            get_streaming_response_api(message, history, api_key, model),
            content_type='text/event-stream'
        )
