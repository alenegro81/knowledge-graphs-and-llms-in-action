import sys

from util.base_importer import BaseImporter


class TopLevelCategoriesPseudoImporter(BaseImporter):

    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "ned-llm"
        with self._driver.session() as session:
            session.run(f"CREATE DATABASE `{self._database}` IF NOT EXISTS")

    def get_rows(self):
        propagation_query = """
        MATCH p=(n:SnomedEntity)<-[:SNOMED_IS_A]-(m:SnomedEntity) 
        WHERE n.id= "138875005" // Root node
        WITH distinct m as first_node
        
        CALL apoc.path.expandConfig(first_node, {
                relationshipFilter: '<SNOMED_IS_A',
                minLevel: 1,
                maxLevel: -1,
                uniqueness: 'RELATIONSHIP_GLOBAL'
            }) yield path
        UNWIND nodes(path) as other_level
        WITH first_node, collect(DISTINCT other_level) as uniques
        
        UNWIND uniques as unique_other_level
        WITH first_node,unique_other_level
        WHERE not first_node.name in coalesce(unique_other_level.type,[])
        
        RETURN unique_other_level.id as id, first_node.name as label
        """

        with self._driver.session(database=self._database) as session:
            result = session.run(query=propagation_query)
            for record in iter(result):
                 yield dict(record)

    def count_rows(self):
        propagation_query = """
        MATCH p=(n:SnomedEntity)<-[:SNOMED_IS_A]-(m:SnomedEntity) 
        WHERE n.id= "138875005" // Root node
        WITH distinct m as first_node

        CALL apoc.path.expandConfig(first_node, {
                relationshipFilter: '<SNOMED_IS_A',
                minLevel: 1,
                maxLevel: -1,
                uniqueness: 'RELATIONSHIP_GLOBAL'
            }) yield path
        UNWIND nodes(path) as other_level
        WITH first_node, collect(DISTINCT other_level) as uniques

        UNWIND uniques as unique_other_level
        WITH first_node,unique_other_level
        WHERE not first_node.name in coalesce(unique_other_level.type,[])

        RETURN count(*) as rows"""

        with self._driver.session(database=self._database) as session:
            return session.run(query=propagation_query).single()["rows"]

    def propagate_categories(self):
        propagate_query = """
        UNWIND $batch as item
        MATCH (e:SnomedEntity)
        WHERE e.id = item.id
        SET e.type = coalesce(e.type, []) + item.label
        """
        size = self.count_rows()
        self.batch_store(propagate_query, self.get_rows(), size=size)


if __name__ == '__main__':
    importing = TopLevelCategoriesPseudoImporter(argv=sys.argv[1:])

    importing.propagate_categories()
    importing.close()
