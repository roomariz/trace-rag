from pymilvus import MilvusClient
client = MilvusClient(uri='tcp://localhost:19530')
results = client.query(
    collection_name='pdf_collection', 
    filter='id >= 0',
    limit=5, 
    output_fields=['id', 'text', 'section', 'page']
)
for r in results:
    print(f"ID: {r['id']}, Page: {r.get('page', '?')}, Section: {r.get('section', '?')[:30] if r.get('section') else '?'}")
    print(f"  Text: {r.get('text', '')[:80]}...")