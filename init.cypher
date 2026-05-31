CREATE VECTOR INDEX entity_description_index IF NOT EXISTS
FOR (n:Event) ON (n.description)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};