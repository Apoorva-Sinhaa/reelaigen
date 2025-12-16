class ContentExtractor:
  
    def __init__(self, content):
        self.content = content

    def extract(self):
  
        return {
            "allText": self.content.allText,
            "noOfSeparations": self.content.noOfSep
        }
