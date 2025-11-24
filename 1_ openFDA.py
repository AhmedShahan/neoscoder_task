from langchain.tools import tool
import requests

@tool
def get_drug_purpose(drug_name: str) -> str:
    """Gets the drug purpose from OpenFDA API."""
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
    response = requests.get(url)

    if response.status_code != 200:
        return f"Error: {response.status_code}"

    data = response.json()
    results = data.get('results', [])
    if not results:
        return "No results found."
    return results
    # # Extract the 'purpose' field
    # purpose = results[0].get('purpose')
    # if purpose:
    #     return "Purpose:\n" + "\n".join(purpose)
    # else:
    #     return "Purpose not found."

# Test it

drug_name="Naproxen"
response = get_drug_purpose.invoke({"drug_name": drug_name})
result = response[0]
# Extract top-level keys
# for key in result.keys():
#     print(key)
#     # If the value is a dictionary, extract its keys too
#     if isinstance(result[key], dict):
#         for sub_key in result[key].keys():
#             print(f"  -> {sub_key}")
#             # Check if the value of sub_key is also a dictionary
#             if isinstance(result[key][sub_key], dict):
#                 for sub_sub_key in result[key][sub_key].keys():
#                     print(f"    --> {sub_sub_key}")


# Example for extracting and printing values
# (Assuming result is a nested dict from your response)
for key in result.keys():
    value = result[key]
    if isinstance(value, dict):
        print(f"{key}:")
        for sub_key in value.keys():
            sub_value = value[sub_key]
            if isinstance(sub_value, dict):
                print(f"  {sub_key}:")
                for sub_sub_key in sub_value.keys():
                    print(f"    {sub_sub_key}: {sub_value[sub_sub_key]}")
            else:
                print(f"  {sub_key}: {sub_value}")
    else:
        print(f"{key}: {value}")
        print("*"*70)

