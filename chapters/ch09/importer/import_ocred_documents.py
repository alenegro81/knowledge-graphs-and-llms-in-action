import json
import sys
from pathlib import Path

from util.base_importer import BaseImporter


class OcredDocumentsImport(BaseImporter):

    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "ned"
        self._size = None

    def set_constraints(self):
        queries = ["CREATE INDEX PageId IF NOT EXISTS FOR (n:Page) ON (n.id)",
                   "CREATE INDEX FileId IF NOT EXISTS FOR (n:File) ON (n.id)",
                   "CREATE FULLTEXT INDEX PageText IF NOT EXISTS FOR (n:Page) ON EACH [n.text]"]

        for q in queries:
            with self._driver.session(database=self._database) as session:
                session.run(q)

    @staticmethod
    def get_page_count(base_path):
        return 28332
        return sum(
            len(json.load(json_file.open())['enriched']['result']['raw_data'])
            for json_file in tqdm(list(base_path.glob("*.json")), desc="computing Page count"))

    @staticmethod
    def manage_hypens(text):
        cleaned_text = ''
        lines = text.splitlines()
        for i, l in enumerate(lines):
            if l[-1] == '-' and i != len(lines) - 1:
                l = l[:len(l) - 1]
            else:
                l += '\n'
            cleaned_text += l

        return cleaned_text

    @classmethod
    def extract_text(cls, blocks, confidence=70):
        left = ''
        right = ''
        for _, item in enumerate(blocks):
            if item['BlockType'] == 'LINE' and item['Confidence'] > confidence:
                line = item['Text'] + '\n'
                if item['Geometry']['BoundingBox']['Left'] < 0.5:
                    left += line
                else:
                    right += line

        text = left + right
        text = cls.manage_hypens(text)

        return text

    @classmethod
    def get_rows(cls, base_path):
        for json_file in base_path.glob("*.json"):
            data = json.load(json_file.open())
            for page in data['enriched']['result']['raw_data']:
                yield {
                    'name': data['file_name'],
                    'type': data['type'],
                    'page': {
                        'page_idx': page['page_number'],
                        'text': cls.extract_text(page['Blocks'])
                    }
                }

    def import_pages(self, base_path):
        page_query = """
        UNWIND $batch as item
        MERGE (f:File {id: item.name})
        SET f.type = item.type, f.path = item.name
        WITH item,f
    
        MERGE (p:Page {id: replace(item.name, '.pdf', '') + '_' + item.page.page_idx})
        SET p.page_idx = item.page.page_idx,
            p.text = item.page.text
        
        MERGE (f)-[:CONTAINS_PAGE]->(p)
        """

        size = self.get_page_count(base_path)
        self.batch_store(page_query, self.get_rows(base_path), size=size)


if __name__ == '__main__':
    importing = OcredDocumentsImport(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/ocred_documents/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    importing.set_constraints()
    importing.import_pages(base_path)
    importing.close()
