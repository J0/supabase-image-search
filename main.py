#!/usr/bin/env python3

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/query")
async def query():
    return {"message": "query"}


@app.post("/query")
async def query():
    return {"message": "query"}
