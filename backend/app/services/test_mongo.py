from pymongo import MongoClient
from pprint import pprint

client = MongoClient("mongodb://localhost:27017")

db = client["novac"]

collection = db["analysis_results"]

docs = list(collection.find())

print(f"Found {len(docs)} document(s)\n")

for doc in docs:
    pprint(doc)
    print("\n" + "="*80 + "\n")