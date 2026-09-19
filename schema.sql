.load .venv/lib/python3.13/site-packages/sqlite_vec/vec0

CREATE TABLE images (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    filename    TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    file_size   INTEGER,
    mtime       INTEGER
);

CREATE VIRTUAL TABLE image_embeddings USING vec0(
    image_id INTEGER PRIMARY KEY,
    embedding FLOAT[512]
);
