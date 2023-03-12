# supabase-neural-search
Reverse image recommendations using FastAPI + Supabase + Huggingface


## What is this?

This project aims to demonstrate how to make use of FastAPI, Supabase, and Huggingface to create an image recommendation system. Users can use this set of API endpoints to upload images to Supabase Storage for training and also make a query for recommendations of similar images. 

Inspired by:

1. [Hugging Face's Article on Image Similarity](https://huggingface.co/blog/image-similarity)_
2. [Hacker News comment](https://news.ycombinator.com/item?id=34967397)
3. [Supabase's Blog post on storing OpenAI Embeddingss](https://supabase.com/blog/openai-embeddings-postgres-vector)


## Motivation

I wanted to play around with PGVector as a means of getting a quick and dirty image recommendation system and as a stepping stone for testing out Postgres Text Search and image recommendation capabilitiyes.

## Getting Started 

### Set up on this repo
1. `virtualenv .env && source .env/bin/activate`
2. `pip3 install -r requirements.txt`
3. `uvicorn main:app --reload`


### Set up on Supabase

You will need to have the `pg_vector` extension enabled. Install this function (taken from blog post)

```sql
CREATE OR REPLACE FUNCTION public.match_documents(query_embedding vector, similarity_threshold double precision, match_count integer)
 RETURNS TABLE(id bigint, content text, similarity double precision)
 LANGUAGE plpgsql
AS $function$
begin
  return query
  select
    documents.id,
    content,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > similarity_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$function$

```



## Endpoints (WIP)

- `/embeddings` 
- `/query`


## Possible Future Extensions


- [ ] Make use of Locality Sensitive Hashing
- [ ] Make use of text embeddings via CLIP to allow image querying based on text
- [ ] Find a loose evaluation metric against other methods of image recommendation.
- [ ] Allow uploads via method other than UI
