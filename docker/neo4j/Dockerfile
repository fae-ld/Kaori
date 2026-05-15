# docker/neo4j/Dockerfile
FROM neo4j:latest

# Kita hanya perlu root sebentar untuk mastiin folder import (backups) 
# bisa diakses bareng sama FastAPI
USER root

# Cukup fokus ke folder import/backups karena ini yang dipakai sharing
RUN mkdir -p /var/lib/neo4j/import && chmod 777 /var/lib/neo4j/import

# Balik ke user default
USER neo4j