import json

def save_to_json(output_data, output_path):
    """Save data to JSON file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


