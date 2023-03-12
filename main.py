#!/usr/bin/env python3

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class File(BaseModel):
    name: str




@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/query/")
async def query(file_: File):
    """
    Returns the storage URL of the top n most similar images to the image uploaded
    """
    return {"message": file_.name}


@app.post("/embeddings")
async def embeddings():
    """
    Generates embeddings from given bucket and stores in tabel.
    """
    return {"message": bucket_path}
