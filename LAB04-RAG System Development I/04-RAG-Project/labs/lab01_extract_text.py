import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.document_loader import load_qa_file # เรียก src/document_loader

def main():
    print("=== Lab 1: Extract Text ===")
    print(f"Reading file: {config.SOURCE_FILE}")

    records = load_qa_file(config.SOURCE_FILE)
    print(f"Found a total of {len(records)} question-answer pairs")

    with open(config.EXTRACTED_TEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {config.EXTRACTED_TEXT_FILE}")

    # Display sample of the first 2 items to verify correct reading
    print("\nSample data read:")
    for record in records[:2]:
        print(f"  - [{record['category']}] Q: {record['question'][:50]}...")

if __name__ == "__main__":
    main()

