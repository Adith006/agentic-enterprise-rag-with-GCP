import re
import logfire
from bs4 import BeautifulSoup


def parse_html_file(file_path):
    """
    Parse an HTML file, clean the extracted text, and return it
    as a single joined string.

    Args:
        file_path (str): Path to the HTML file.

    Returns:
        str | None: Cleaned text, or None if parsing fails.
    """
    try:
        with logfire.span("Parsing and cleaning HTML file", file_path=file_path):

            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()

            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unwanted HTML elements
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            # Extract text
            text = soup.get_text(separator=" ")

            # Clean junk/whitespace
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"[^\x00-\x7F]+", " ", text)

            # Join cleaned text
            cleaned_text = " ".join(text.split())

            return cleaned_text

    except FileNotFoundError as e:
        logfire.error("HTML file not found: {error}", error=str(e))
        return None

    except Exception as e:
        logfire.error("Failed to parse HTML file: {error}", error=str(e))
        return None