#!/usr/bin/env python3

from pathlib import Path
import sqlite3
import sys

import numpy as np
import sqlite_vec
import torch
import open_clip


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DATABASE = Path("./images.sqlite")

# MUST be identical to the model used by indexate.py
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# No NVIDIA/CUDA required.
DEVICE = "cpu"


# ----------------------------------------------------------------------
# Load CLIP model
# ----------------------------------------------------------------------

print("Loading model...", file=sys.stderr)

model, _, _ = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    device=DEVICE,
)

tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.eval()


# ----------------------------------------------------------------------
# Open SQLite and load sqlite-vec
# ----------------------------------------------------------------------

db = sqlite3.connect(DATABASE)

db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)


# ----------------------------------------------------------------------
# Convert text to CLIP embedding
# ----------------------------------------------------------------------

def encode_query(text):
    tokens = tokenizer([text])
    with torch.no_grad():
        vector = model.encode_text(tokens)
    # Same normalization used for the image embeddings.
    vector = vector / vector.norm(dim=-1, keepdim=True)
    return vector.cpu().numpy()[0].astype(np.float32)


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------

def vector_search(query_vector, limit=20):
    # sqlite-vec expects the vector as bytes.
    query_blob = query_vector.tobytes()
    rows = db.execute(
        """
        SELECT
            image_id,
            distance
        FROM image_embeddings
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
        """,
        (query_blob, limit),
    ).fetchall()
    return rows


# ----------------------------------------------------------------------
# Get image paths
# ----------------------------------------------------------------------

def get_paths(results):
    paths = []
    for image_id, distance in results:
        row = db.execute(
            """
            SELECT path
            FROM images
            WHERE id = ?
            """,
            (image_id,),
        ).fetchone()
        if row is not None:
            paths.append((row[0], distance))
    return paths


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # Query comes from stdin.
    query = input('Enter query: ') #sys.stdin.read().strip()
    if not query:
        print("No query supplied.", file=sys.stderr)
        sys.exit(1)
    print(f"Query: {query}", file=sys.stderr)
    query_vector = encode_query(query)
    results = vector_search(query_vector, limit=20)
    paths = get_paths(results)
    for path, distance in paths:
        print(f"{distance:.6f}\t{path}")


if __name__ == "__main__":
    main()
