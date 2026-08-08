import re


class ClauseSegmenter:
    """
    Segments legal contracts into clause-level units.

    Detects:
    - Numbered clauses (1., 1.1, 2.3.4)
    - ARTICLE headings
    - SECTION headings

    Preserves:
    - Document name
    - Page range
    - Clause number
    - Clause title
    - Clause text
    """

    def __init__(self):

        self.number_pattern = re.compile(
            r"^\d+(\.\d+)*\.?\s*(.*)$"
        )

        self.article_pattern = re.compile(
            r"^ARTICLE\s+[IVXLC]+",
            re.IGNORECASE
        )

        self.section_pattern = re.compile(
            r"^SECTION\s+\d+(\.\d+)*",
            re.IGNORECASE
        )

    def is_heading(self, line):
        """
        Returns True if a line is a legal clause heading.
        """

        return (
            self.number_pattern.match(line)
            or self.article_pattern.match(line)
            or self.section_pattern.match(line)
        )

    def build_clause(self, document_name, page_number, heading):
        """
        Creates a new clause dictionary.
        """

        clause_number = ""
        title = heading

        match = self.number_pattern.match(heading)

        if match:
            clause_number = heading.split()[0]
            title = heading[len(clause_number):].strip()

        return {
            "document": document_name,
            "page_start": page_number,
            "page_end": page_number,
            "clause_number": clause_number,
            "title": title,
            "text": ""
        }

    def segment(self, document):
        """
        Splits an extracted document into legal clauses.
        """

        clauses = []
        current_clause = None

        for page in document["pages"]:

            page_number = page["page"]

            # If a clause spans multiple pages,
            # keep updating its ending page.
            if current_clause is not None:
                current_clause["page_end"] = page_number

            lines = page["text"].splitlines()

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                if self.is_heading(line):

                    # Save previous clause
                    if current_clause is not None:

                        current_clause["text"] = current_clause["text"].strip()

                        if current_clause["text"]:
                            clauses.append(current_clause)

                    # Start new clause
                    current_clause = self.build_clause(
                        document["filename"],
                        page_number,
                        line
                    )

                else:

                    # Everything before the first heading
                    # becomes the preamble.
                    if current_clause is None:

                        current_clause = {
                            "document": document["filename"],
                            "page_start": page_number,
                            "page_end": page_number,
                            "clause_number": "",
                            "title": "Preamble",
                            "text": ""
                        }

                    current_clause["text"] += line + "\n"

        # Save last clause
        if current_clause is not None:

            current_clause["text"] = current_clause["text"].strip()

            if current_clause["text"]:
                clauses.append(current_clause)

        return clauses