

# Read text from a file
def read_file_local(file_path):
    raw_text = ""
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()
        
    return raw_text