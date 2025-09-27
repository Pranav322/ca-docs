#!/usr/bin/env python3
"""
Explore CA folder structure and show how metadata will be inferred
"""

import os
from pathlib import Path

def explore_ca_folder(ca_path="ca"):
    """
    Explore CA folder and show how metadata will be inferred
    """
    ca_folder = Path(ca_path)
    
    if not ca_folder.exists():
        print(f"❌ CA folder not found: {ca_folder.resolve()}")
        print(f"Please create the folder and organize your PDFs like this:")
        print(f"ca/")
        print(f"├── Foundation/")
        print(f"│   ├── Paper 1/")
        print(f"│   │   ├── Module A/")
        print(f"│   │   │   └── Chapter 1.pdf")
        print(f"│   │   └── some-file.pdf")
        print(f"│   └── Paper 2/")
        print(f"├── Intermediate/")
        print(f"│   └── Group I Paper 1/")
        print(f"└── Final/")
        print(f"    └── Group I Paper 1/")
        return
    
    print(f"📁 Exploring CA folder: {ca_folder.resolve()}")
    print()
    
    # Function to infer metadata like the batch script does
    def build_metadata(path_parts, file_path):
        def normalize(name):
            return name.strip().replace(" .", ".").replace("..", ".").replace("  ", " ")
        
        meta = {
            "level": None,
            "paper": None, 
            "module": None,
            "chapter": None,
            "unit": None,
            "source_file": file_path,
            "file_name": path_parts[-1] if path_parts else "unknown.pdf"
        }
        
        # Extract hierarchy from path
        if len(path_parts) >= 1:
            meta["level"] = normalize(path_parts[0])
        if len(path_parts) >= 2:
            meta["paper"] = normalize(path_parts[1])
        if len(path_parts) >= 3 and path_parts[2].lower().startswith("module"):
            meta["module"] = normalize(path_parts[2])
        if len(path_parts) >= 4 and path_parts[3].lower().startswith("chapter"):
            meta["chapter"] = normalize(path_parts[3])
        
        # Check filename for unit/chapter info
        filename = path_parts[-1]
        if filename.lower().startswith("unit"):
            meta["unit"] = normalize(filename.replace(".pdf", ""))
        elif filename.lower().startswith("chapter"):
            meta["chapter"] = normalize(filename.replace(".pdf", ""))
        
        return meta
    
    # Collect all PDFs and their inferred metadata
    pdfs_found = []
    
    for pdf_file in ca_folder.rglob("*.pdf"):
        rel_path = pdf_file.relative_to(ca_folder)
        path_parts = rel_path.parts
        metadata = build_metadata(path_parts, str(rel_path))
        
        pdfs_found.append({
            'file': pdf_file,
            'rel_path': rel_path,
            'metadata': metadata
        })
    
    if not pdfs_found:
        print("❌ No PDF files found in the CA folder")
        print("\nPlease add your CA study material PDFs to the folder structure")
        return
    
    print(f"✅ Found {len(pdfs_found)} PDF files:")
    print()
    
    # Show folder structure
    print("📂 Folder Structure:")
    print("=" * 50)
    
    # Group by level
    levels = {}
    for pdf_info in pdfs_found:
        level = pdf_info['metadata']['level'] or 'Unknown'
        if level not in levels:
            levels[level] = {}
        
        paper = pdf_info['metadata']['paper'] or 'Unknown'
        if paper not in levels[level]:
            levels[level][paper] = []
        
        levels[level][paper].append(pdf_info)
    
    # Display structure
    for level_name, papers in levels.items():
        print(f"📚 {level_name}/")
        for paper_name, files in papers.items():
            print(f"  📄 {paper_name}/")
            for file_info in files[:3]:  # Show first 3 files
                file_name = file_info['file'].name
                metadata = file_info['metadata']
                print(f"    📋 {file_name}")
                if metadata['module']:
                    print(f"        Module: {metadata['module']}")
                if metadata['chapter']:
                    print(f"        Chapter: {metadata['chapter']}")
                if metadata['unit']:
                    print(f"        Unit: {metadata['unit']}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more files")
        print()
    
    # Show validation results
    print("🔍 Metadata Validation:")
    print("=" * 50)
    
    valid_files = []
    invalid_files = []
    
    for pdf_info in pdfs_found:
        metadata = pdf_info['metadata']
        
        # Check basic requirements and fix level capitalization
        is_valid = metadata['level'] and metadata['paper']
        
        if is_valid and metadata['level']:
            # Case-insensitive level validation and fix capitalization
            valid_levels = ['Foundation', 'Intermediate', 'Final']
            level_found = False
            for valid_level in valid_levels:
                if metadata['level'].lower() == valid_level.lower():
                    metadata['level'] = valid_level  # Fix capitalization
                    level_found = True
                    break
            is_valid = level_found
        
        if is_valid:
            valid_files.append(pdf_info)
        else:
            invalid_files.append(pdf_info)
    
    print(f"✅ Valid files: {len(valid_files)}")
    print(f"❌ Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print("\n❌ Files with invalid metadata (will be skipped):")
        for pdf_info in invalid_files:
            print(f"  - {pdf_info['rel_path']}")
            print(f"    Metadata: {pdf_info['metadata']}")
    
    print()
    print("🚀 Ready for batch processing!")
    print(f"  - Total files: {len(pdfs_found)}")
    print(f"  - Valid files: {len(valid_files)}")
    print(f"  - Will be processed: {len(valid_files)}")

if __name__ == "__main__":
    explore_ca_folder()
