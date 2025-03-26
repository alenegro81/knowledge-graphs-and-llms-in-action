import math
from itertools import islice
from typing import Iterable

from util.graphdb_base import GraphDBBase
from tqdm import tqdm


class BaseImporter(GraphDBBase):
    def __init__(self, command=None, argv=None, extended_options='', extended_long_options=None):
        super().__init__(command, argv, extended_options, extended_long_options)
        self._database = "neo4j"
        self.batch_size = 1000

    def batch_store(self, query: str, parameters_iterator: Iterable, size: int = None, strategy: str = "aggregate",
                    desc=""):
        """
        Ingest data in batches
        :seeAlso transaction_batch_store
        :seeAlso aggregate_batch_store

        :param query: the parametrized insertion query
        :param parameters_iterator: an iterator of the data to ingest as parameters for query
        :param size: optional parameters_iterator's length
        :param strategy: "aggregate" or "transaction"
        :param desc: optional progress bar description
        """
        method = getattr(self, f"{strategy}_batch_store", None)
        if method is None:
            raise ValueError(f"Unknown strategy {strategy}")
        method(query, parameters_iterator, size, desc)

    def transaction_batch_store(self, query, parameters_iterator, size=None, desc=""):
        """
        Ingest data submitting a query for every item in parameters_iterator and commit every 1000 iterations
        :param query: the parametrized insertion query
        :param parameters_iterator: an iterator of the data to ingest as parameters for query
        :param size:optional parameters_iterator's length
        :param desc: optional progress bar description
        """
        parameters_iterator = tqdm(parameters_iterator, total=size, desc=desc)
        with self._driver.session(database=self._database) as session:
            tx = session.begin_transaction()
            for item_count, parameters in enumerate(parameters_iterator, start=1):
                tx.run(query, parameters)
                if item_count % self.batch_size == 0:
                    tx.commit()
                    tx = session.begin_transaction()
            tx.commit()

    @staticmethod
    def get_csv_size(HMDD_file, encoding="utf-8"):
        return sum(1 for i in HMDD_file.open("r", encoding=encoding))

    @staticmethod
    def get_batches(parameters_iterator, batch_size):
        while True:
            ret = list(islice(parameters_iterator, batch_size))
            if ret:
                yield ret
            else:
                return

    def aggregate_batch_store(self, query, parameters_iterator, size=None, desc=""):
        """
        Ingest data in batches
          It aggregates parameters_iteration in a list named $batch,
          the query should contain `UNWIND $batch as item` as first statement
        :param query: the parametrized insertion query
        :param parameters_iterator: an iterator of the data to ingest as parameters for query
        :param size:optional parameters_iterator's length
        :param desc: optional progress bar description
        """
        parameters_batches = self.get_batches(parameters_iterator, self.batch_size)
        parameters_batches = tqdm(parameters_batches, total=math.ceil(size / self.batch_size), desc=desc)
        with self._driver.session(database=self._database) as session:
            for batch in parameters_batches:
                session.run(query, {"batch": batch})
