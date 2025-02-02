#!/usr/bin/env python3
import argparse
import json
import random
import string
from datetime import datetime, timedelta

def random_timestamp(start, end):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=random_seconds)).strftime("%Y-%m-%d %H:%M:%S")

def generate_document(doc_id: int) -> dict:
    # For variety, randomly choose a bill type
    bill_types = ["Electricity Bill", "Water Bill", "Gas Bill", "Internet Bill", "Telephone Bill"]
    bill_type = random.choice(bill_types)
    name = f"{bill_type} #{doc_id}"
    
    # Generate a synthetic large content by repeating a sample paragraph with slight variations
    sample_paragraph = (
        f"{bill_type} Details:\n"
        "Customer Name: ilker e\n"
        f"Account Number: {random.randint(100000000, 999999999)}\n"
        "Billing Period: 2025-06-01 to 2025-06-30\n"
        f"Total Consumption: {random.randint(100, 500)} units\n"
        f"Amount Due: EUR {round(random.uniform(30.0, 300.0), 2)}\n"
        "Due Date: 2025-07-15\n"
        "Details:\n"
        "- Fixed Charge: EUR 10.00\n"
        "- Variable Charge: EUR " + str(round(random.uniform(20.0, 290.0), 2)) + "\n"
        "- Taxes and Fees: EUR " + str(round(random.uniform(1.0, 10.0), 2)) + "\n"
        "\n"
    )
    repetitions = random.randint(5, 15)
    content = sample_paragraph * repetitions

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    timestamp = random_timestamp(start_date, end_date)

    url = f"http://example.com/documents/{doc_id}"

    return {
        "name": name,
        "content": content,
        "timestamp": timestamp,
        "url": url
    }

def generate_documents(num_docs: int, output_file: str):
    with open(output_file, "w") as f:
        for doc_id in range(1, num_docs + 1):
            doc = generate_document(doc_id)
            f.write(json.dumps(doc) + "\n")
    print(f"Generated {num_docs} documents in {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a large JSON Lines file with synthetic documents."
    )
    parser.add_argument("--num-docs", type=int, default=1000,
                        help="Number of documents to generate (default: 1000)")
    parser.add_argument("--output", type=str, default="documents.jsonl",
                        help="Output JSON Lines file (default: documents.jsonl)")
    args = parser.parse_args()
    
    generate_documents(args.num_docs, args.output)

if __name__ == "__main__":
    main()

