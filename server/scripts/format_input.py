import json
import re

# Load the JSON file
with open("/Users/sibaprasadtripathy/Downloads/business_schools_bulk.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Clean up total_course_fees
for record in data:
    if "total_course_fees" in record:
        record["total_course_fees"] = float(re.sub(r"[^\d.]", "", record["total_course_fees"]))

# Save the cleaned data
with open("/Users/sibaprasadtripathy/Downloads/business_schools_clean.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)

