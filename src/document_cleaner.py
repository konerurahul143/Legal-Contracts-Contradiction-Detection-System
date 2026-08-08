import re


class DocumentCleaner:
    """
    Cleans raw legal document text before clause segmentation.

    Removes:
    - Repeated page headers
    - Page footers
    - SEC filing metadata
    - Exhibit labels
    - Extra whitespace

    Keeps all legal content unchanged.
    """

    def __init__(self):

        self.patterns = [

            # SEC source lines
            re.compile(
                r"^Source:.*$",
                re.IGNORECASE
            ),

            # Exhibit labels
            re.compile(
                r"^Exhibit\s+\S+.*$",
                re.IGNORECASE
            ),

            # Page numbers
            re.compile(
                r"^Page\s+\d+.*$",
                re.IGNORECASE
            ),

            # Standalone page numbers
            re.compile(
                r"^\d+$"
            ),

            # --------- Add more later if needed ---------
        ]

    def clean_line(self, line):

        line = line.strip()

        if not line:
            return ""

        for pattern in self.patterns:

            if pattern.match(line):
                return ""

        return line

    def clean_document(self, document):

        cleaned_pages = []

        for page in document["pages"]:

            cleaned_lines = []

            for line in page["text"].splitlines():

                line = self.clean_line(line)

                if line:
                    cleaned_lines.append(line)

            cleaned_pages.append({
                "page": page["page"],
                "text": "\n".join(cleaned_lines)
            })

        return {
            "filename": document["filename"],
            "pages": cleaned_pages
        }