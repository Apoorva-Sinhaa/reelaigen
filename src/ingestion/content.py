from unstructured.partition.auto import partition
class Content:
    def __init__(self, file_path: str):
        self.allText = ""
        self.noOfSep = 0

        self._ingest(file_path)

    def _ingest(self, file_path: str):
        elements = partition(filename=file_path)
        self.allText = "\n\n".join(
            el.text for el in elements if el.text
        )
        self.noOfSep = len(elements)

        if not self.allText.strip():
            print("[WARN] No extractable text found.")
