from pymilvus import MilvusClient
client = MilvusClient(uri='tcp://localhost:19530')
results = client.query(collection_name='pdf_collection', filter='id >= 0', limit=20, output_fields=['id', 'section', 'subsection'])
for r in results:
    print(f'ID {r["id"]} section={r.get("section","")[:20]} subsection={r.get("subsection","")[:30]}')