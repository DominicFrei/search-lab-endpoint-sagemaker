import json
import os

import boto3
import pymongo
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")

sagemaker_runtime_client = boto3.client("sagemaker-runtime",
                                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                        region_name=AWS_REGION_NAME)


def get_embedding(synopsis):
    input_data = {"text_inputs": synopsis}

    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName="jumpstart-dft-hf-textembedding-all-minilm-l6-v2",
        Body=json.dumps(input_data),
        ContentType="application/json"
    )
    result = json.loads(response["Body"].read().decode())

    embedding = result["embedding"][0]

    return embedding


def process_and_save_books():
    mongo_client = pymongo.MongoClient(CONNECTION_STRING)
    library_database = mongo_client["library"]
    books_collection = library_database["books"]
    books = list(books_collection.find())

    for book in books:
        synopsis = book["synopsis"]
        embedding = get_embedding(synopsis)
        book["embeddings"] = embedding

    sagemaker_books_collection = library_database["sagemakerBooks"]
    sagemaker_books_collection.drop()
    sagemaker_books_collection.insert_many(books)


if __name__ == "__main__":
    process_and_save_books()
