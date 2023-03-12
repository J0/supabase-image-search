# supabase-neural-search
Reverse image recommendations using FastAPI + Supabase + Huggingface


## What is this?

This project aims to demonstrate how to make use of FastAPI, Supabase, and Huggingface to create an image recommendation system. Users can use this set of API endpoints to upload images to Supabase Storage for training and also make a query for recommendations of similar images. 

Inspired by:

1. [Hugging Face's Article on Image Similarity](https://huggingface.co/blog/image-similarity)_
2. [Hacker News comment](https://news.ycombinator.com/item?id=34967397)


## Motivation

I wanted to play around with PGVector as a means of getting a quick and dirty image recommendation system and as a stepping stone for testing out Postgres Text Search and image recommendation capabilitiyes.


## Endpoints (WIP)

- `generate_embeddings` 
- `query_image`


## Possible Future Extensions


- [ ] Make use of Locality Sensitive Hashing
- [ ] Make use of text embeddings via CLIP to allow image querying based on text
- [ ] Find a loose evaluation metric against other methods of image recommendation.
