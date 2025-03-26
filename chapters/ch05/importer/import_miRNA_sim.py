import sys
from pathlib import Path

import numpy as np
import pandas as pd

from util.base_importer import BaseImporter


class MatrixSimilarityImporter(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "hmdd2.0"

    def getMatrixSize(self, similarity_file, threshold):
        matrix = np.loadtxt(similarity_file)
        # count the number of items exceeding the threshold in a flatted version of the matrix
        # (True counts as 1, False counts as 0)
        # only lower  lower-triangle without counting the 1s on diagolan
        return sum(np.tril(matrix).flatten() > threshold) - matrix.shape[0]

    def get_rows(self, names_file, similarity_file, threshold):
        names = pd.read_excel(names_file, header=None)[0].tolist()
        matrix = np.loadtxt(similarity_file)
        assert matrix.shape == (len(names), len(names)), "names count and adjiacenty matrix mismatch"
        for i, name_src in enumerate(names):
            for j, name_dst in enumerate(names):
                if j < i:
                    similarity_value = matrix[i][j]
                    if similarity_value > threshold:
                        yield {"sourceName": name_src.lower(),
                               "destinationName": name_dst.lower(),
                               "value": float(similarity_value)}

    def import_similarity_matrix(self, names_file, similarity_file, threshold=0):
        query = """
                UNWIND $batch as item
                MATCH (source:MiRNA {name: item.sourceName})
                MATCH (destination:MiRNA {name: item.destinationName})
                MERGE (source)-[r:SIMILAR_TO ]->(destination)
                SET r.value = item.value
            """
        size = self.getMatrixSize(similarity_file, threshold)
        self.batch_store(query, self.get_rows(names_file, similarity_file, threshold), size=size)


def main():
    importing = MatrixSimilarityImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/hmdd/misim"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    names = base_path / "microRNA.xls"
    similarityMatrix = base_path / "similarityMatrix.txt"

    if not names.is_file():
        print(names, "doesn't exist in ", base_path)
        sys.exit(1)

    if not similarityMatrix.is_file():
        print(similarityMatrix, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.import_similarity_matrix(names_file=names, similarity_file=similarityMatrix, threshold=0.25)
    importing.close()


if __name__ == '__main__':
    main()
