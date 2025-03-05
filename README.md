# Interactive AI Resume Chat


<div align="center">
  
🚀 [Try it out yourself!](http://danielsbeiticv.com){:target="_blank"}
  
</div>


## Home Interface
<img width="1702" alt="Screenshot 2025-03-04 at 23 39 42" src="https://github.com/user-attachments/assets/ef9a85d5-f2d0-4f7a-9f4d-81bd1b8382b9" style="border: 2px solid white;" />

## Chatbot Interface
<img width="1702" alt="Screenshot 2025-03-04 at 23 39 26" src="https://github.com/user-attachments/assets/ca450986-e6d8-4a25-9ba1-4caa6568aa53" style="border: 2px solid white;" />



## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Future Improvements](#future-improvements)

## Features

- **Interactive Chat Interface**: Engage with a conversational AI about my professional experience
- **PDF Resume Viewer**: View or download my resume directly from the application
- **Streaming Responses**: Enjoy real-time streaming responses for a natural conversation flow
- **Conversation Memory**: Chat history is maintained for contextual follow-up questions
- **Dark Mode UI**: Elegant dark-themed interface for comfortable viewing
- **Responsive Design**: Works across desktop and mobile devices

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Backend**: Python (Django/Flask/FastAPI)
- **AI Model**: Large Language Model with RAG architecture
- **Vector Database**: For efficient retrieval of relevant resume sections
- **Streaming**: Server-Sent Events for real-time response generation

## Architecture

The application follows a Retrieval-Augmented Generation (RAG) architecture:

1. **Document Processing**:
   - My resume is parsed and divided into meaningful chunks
   - Each chunk is converted into vector embeddings using an embedding model
   - These embeddings are stored in a vector database for quick retrieval

2. **Query Processing**:
   - User questions are converted to the same embedding space
   - Semantic search identifies the most relevant resume sections
   - Retrieved sections are used as context for the language model

3. **Response Generation**:
   - The language model generates responses based on retrieved context
   - Previous conversation history provides additional context
   - Responses are streamed back to the user for immediate feedback

## How It Works

1. **Resume Indexing**:
   - The system pre-processes my resume and creates searchable embeddings
   - Information is organized into categories like skills, experience, education, etc.

2. **Query Understanding**:
   - When a user asks a question, the system determines the intent
   - It identifies which parts of my resume are relevant to the query

3. **Contextual Retrieval**:
   - The system retrieves specific sections from my resume that answer the query
   - It combines this information with conversation history for context

4. **Response Generation**:
   - The AI generates a natural language response based on the retrieved information
   - The response is specifically crafted to answer the user's question
   - Technical terminology is explained in a way that's easy to understand



## Usage

1. Visit the home page to view my resume
2. Click "Chat with my Resume" to start a conversation
3. Ask questions about my:
   - Work experience (e.g., "Tell me about your role at EDF Singapore")
   - Technical skills (e.g., "What programming languages do you know?")
   - Education background (e.g., "Where did you study?")
   - Projects (e.g., "What was your most challenging project?")
   - General professional information (e.g., "What are your career goals?")

## Implementation Details

### Vector Database

The application uses a vector database to store embeddings of my resume sections. This enables semantic search functionality where the system can match queries with contextually relevant information, even if there's no exact keyword match.

### Prompt Engineering

Custom prompts help the AI model understand:
- When to admit it doesn't know something vs. making inferences
- How to format responses for readability (bullet points, emphasis, etc.)
- How to maintain a professional yet conversational tone
- How to appropriately cite sections of my resume

### Context Window Management

A sliding context window approach helps maintain conversation history while staying within token limits. This ensures the AI has relevant context without exceeding model capacity.

### Performance Optimization

- **Caching**: Frequently asked questions are cached for faster response times
- **Streaming**: Responses are streamed for better user experience
- **Chunk Size Optimization**: Resume data is chunked for optimal retrieval relevance

## Future Improvements

- **Multi-document Support**: Expand beyond resume to include portfolio projects
- **Voice Interface**: Add speech recognition and synthesis capabilities
- **Analytics Dashboard**: Track common questions to improve the resume
- **Multilingual Support**: Answer questions in multiple languages
- **Custom Knowledge Base**: Allow uploading of additional documents for context

---

Built with ❤️ by Daniel Sbeiti
