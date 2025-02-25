import logging
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from rag.generation import get_streaming_response

def home(request):
    return render(request, 'home.html')

def chat(request):
    return render(request, 'chat.html')

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '')
        history = data.get('history', [])  # Get conversation history from request
        return StreamingHttpResponse(
            get_streaming_response(message, history),
            content_type='text/event-stream'
        )
