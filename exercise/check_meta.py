from pymilvus import MilvusClient
client = MilvusClient(uri='tcp://localhost:19530')
results = client.query(collection_name='pdf_collection', limit=3, output_fields=['page', 'section', 'subsection'])
for r in results:
    print(r)