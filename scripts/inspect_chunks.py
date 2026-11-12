from pathlib import Path

from app.documents.chunker import (
    chunk_document,
    load_bge_tokenizer,
)
from app.documents.pdf_loader import load_pdf

pdf_files = list(Path("data/library").glob("*.pdf"))

if len(pdf_files) != 1:
    raise SystemExit(f"Expected exactly one PDF, found {len(pdf_files)}")

document = load_pdf(pdf_files[0])

tokenizer = load_bge_tokenizer()

chunks = chunk_document(
    document=document,
    tokenizer=tokenizer,
    chunk_size=300,
    overlap=50,
)

print("\nDOCUMENT:", document.filename)
print("TOTAL PAGES:", document.page_count)
print("TOTAL CHUNKS:", len(chunks))

for chunk in chunks[:5]:
    print("\n" + "=" * 60)
    print("CHUNK ID:", chunk.chunk_id)
    print("PAGE:", chunk.page_number)
    print("CHUNK INDEX:", chunk.chunk_index)
    print("TOKEN COUNT:", chunk.token_count)
    print("\nTEXT PREVIEW:\n")
    print(chunk.text[:500])
