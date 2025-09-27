import os
import uuid
import json
import pdfplumber

# expand ~ properly
BASE_PATH = os.path.expanduser("~/Documents/ca")

def normalize(name):
    return name.strip().replace(" .", ".").replace("..", ".").replace("  ", " ")

def build_metadata(path_parts, file_path):
    meta = {
        "level": None,
        "paper": None,
        "module": None,
        "chapter": None,
        "unit": None,
        "source_file": file_path
    }

    if len(path_parts) >= 1:
        meta["level"] = normalize(path_parts[0])
    if len(path_parts) >= 2:
        meta["paper"] = normalize(path_parts[1])
    if len(path_parts) >= 3 and path_parts[2].lower().startswith("module"):
        meta["module"] = normalize(path_parts[2])
    if len(path_parts) >= 4 and path_parts[3].lower().startswith("chapter"):
        meta["chapter"] = normalize(path_parts[3])

    filename = path_parts[-1]
    if filename.lower().startswith("unit"):
        meta["unit"] = normalize(filename.replace(".pdf", ""))
    elif filename.lower().startswith("chapter"):
        meta["chapter"] = normalize(filename.replace(".pdf", ""))

    return meta

def extract_text_and_tables(pdf_path):
    full_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    full_content.append(text)

                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    # convert table to Markdown
                    header = "| " + " | ".join(table[0]) + " |"
                    separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
                    rows = ["| " + " | ".join(row) + " |" for row in table[1:]]
                    table_md = "\n".join([header, separator] + rows)
                    full_content.append(table_md)
    except Exception as e:
        print(f"[ERROR] Failed to parse {pdf_path}: {e}")
    return "\n\n".join(full_content)

def crawl(base_path):
    all_metadata = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(".pdf"):
                file_path = os.path.join(root, f)
                rel_path = os.path.relpath(file_path, base_path)
                path_parts = rel_path.split(os.sep)

                meta = build_metadata(path_parts, rel_path)
                content = extract_text_and_tables(file_path)

                all_metadata.append({
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "metadata": meta
                })
    return all_metadata

if __name__ == "__main__":
    data = crawl(BASE_PATH)
    out_file = "ca_metadata.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(data)} entries with text+tables to {out_file}")
