class CandidateSelection:
    # TODO: Add embedding search
    def __init__(self, store):
        self.store = store
        self.database = "ned-llm"

    def full_text_query(self):
        query = """
                CALL db.index.fulltext.queryNodes("snomedEntityName", $fulltextQuery, {limit: $limit})
                YIELD node, score
                WHERE node:SnomedEntity AND ANY(x IN node.type WHERE x IN $labels)
                RETURN distinct node.name AS candidate_name, node.id AS candidate_id
                """

        return query

    def generate_full_text_query(self, input):
        full_text_query = ""
        words = [el for el in input.split() if el]

        if len(words) > 1:
            for word in words[:-1]:
                full_text_query += f" {word}~0.80 AND "
                full_text_query += f" {words[-1]}~0.80"
        else:
            full_text_query = words[0] + "~0.80"

        return full_text_query.strip()

    def get_candidates(self, input, labels, limit = 10):
        ft_query = self.generate_full_text_query(input)
        with self.store._driver.session(database=self.database) as session:
            candidates = session.run(self.full_text_query(), {"fulltextQuery": ft_query, "labels": labels, "limit":limit})

            return [{"snomed_id" :c["candidate_id"], "name": c['candidate_name']} for c in candidates]
