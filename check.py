import re
import json

js_file = "data.js"

def extract_js_array(var_name, js_content):
    """Extract the JS array assigned to var_name."""
    pattern = rf"{var_name}\s*=\s*(\[[\s\S]*?\]);"
    match = re.search(pattern, js_content)
    if match:
        return match.group(1)
    else:
        return None

def check_js_array(js_array_str, var_name):
    """Check if JS array can be converted to Python/JSON."""
    try:
        # Replace JS-style single quotes with double quotes and remove trailing commas
        cleaned = re.sub(r"'", '"', js_array_str)
        cleaned = re.sub(r",(\s*[\]\}])", r"\1", cleaned)  # remove trailing commas
        # Convert to Python object
        data = json.loads(cleaned)
        if isinstance(data, list) and all(isinstance(p, dict) for p in data):
            print(f"{var_name}: OK, {len(data)} points")
        else:
            print(f"{var_name}: INVALID structure")
    except Exception as e:
        print(f"{var_name}: ERROR - {e}")

with open(js_file, "r", encoding="utf-8") as f:
    content = f.read()

for var in ["data40", "data200", "data500"]:
    arr = extract_js_array(var, content)
    if arr:
        check_js_array(arr, var)
    else:
        print(f"{var}: not found or cannot extract")
