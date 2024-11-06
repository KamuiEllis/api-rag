import json
import time
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import glob
import asyncio
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from fastapi import FastAPI, File, UploadFile
import shutil
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import uuid
from pydantic import BaseModel



PINECONE_API_KEY=""
api_key = ''

pc = Pinecone(api_key=PINECONE_API_KEY)
ai = OpenAI(api_key=api_key)


app = FastAPI()

origins = [     # Any other specific origin
    "*"                       # Allow all origins (use with caution)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class Message(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    content: str
    conversation: List[Message]

def getData(query):
    data = pc.Index("data")
    
    embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=query,
    parameters={
        "input_type": "query"
    }
    )

    search_results = data.query(
        vector=embeddings[0]['values'],
        namespace="ns1",
        top_k=10,
        include_values=False,
        include_metadata=True,
    )

    results = [{"id": match["id"], "text": match["metadata"]["text"]}
        for match in search_results['matches']
        if 'text' in match['metadata']]
    
    return results[0]["text"]



@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/send")
async def send(query:Query):
    print(query)
    infoFunction = {
    "name": "getInformation",
    "description": "Call this function when you want to get information related to the user's question",
     "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A question created by the model to get information from a database. E.g Green house gases",
            },
        },
        "required": ["query"],
        "additionalProperties": False
    }
    }

    tools = [{"type": "function", "function": infoFunction}]
    messages = query.conversation
    response = ai.chat.completions.create(model="gpt-4o-2024-05-13", messages=messages, tools=tools, max_tokens=100)

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool = message.tool_calls[0]
        arguments = json.loads(tool.function.arguments)
        returnedData = getData(**arguments)
        newDict = {
            "role": 'tool',
            "tool_call_id": tool.id,
            "content": json.dumps({"question":arguments.get('query'), "answer":returnedData}),
        }
        messages.append(message)
        messages.append(newDict)
        response = ai.chat.completions.create(model="gpt-4o-2024-05-13", messages=messages, max_tokens=100)
        messages.append({"role": 'assistant', "content":response.choices[0].message.content })

    # data = getData(query.content)
    # print(data)
    return {"valid": True, "message": {"role": 'assistant', "content":response.choices[0].message.content }}

@app.post("/upload")
async def uploadFile(file: UploadFile = File(...)):
    try:
        pages = []
        chunks = []
        with open(f"uploads/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader((f"uploads/{file.filename}"))  

        async for page in loader.alazy_load():
            pagedData = page
            pages.append(pagedData)  

        textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = textSplitter.split_documents(pages)  

        toBeVectorized = []
        for x in chunks:
            toBeVectorized.append({"id": str(uuid.uuid4())
, 'text': x.page_content })

        vectorEmbeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[d['text'] for d in toBeVectorized],
            parameters={"input_type": "passage", "truncate": "END"}
        )

        while not pc.describe_index('data').status['ready']:
            time.sleep(1)

        index = pc.Index("data")
        vectors = []

        for d, e in zip(toBeVectorized, vectorEmbeddings):
            vectors.append({
                "id": d['id'],
                "values": e['values'],
                "metadata": {'text': d['text'], 'source': f"uploads/{file.filename}"}
            })

        index.upsert(
            vectors=vectors,
            namespace="ns1"
        )    

        return {"valid": True, "chunks": chunks}
    except Exception as e:
        print(e)
        return {"valid": False}
       

