import logfire


def parse_text_file(file_path):
    """
    Parse a text file and return its contents.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Contents of the text file.
    """
    try:
        with logfire.span("Parsing text file", file_path=file_path):
            with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
                return file.read()

    except FileNotFoundError as e:
        logfire.error("File not found: {error}", error=str(e))
        return None

    except Exception as e:
        logfire.error("Failed to parse text file: {error}", error=str(e))
        return None