from pathlib import Path
import sqlite3
import sqlite_vec
import struct

import torch
import open_clip
from PIL import Image


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

IMAGE_DIR = Path("./images")
DATABASE = Path("./images.sqlite")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Use CUDA if available, otherwise CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------

print(f"Loading model on {DEVICE}...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    device=DEVICE,
)

model.eval()


# ----------------------------------------------------------------------
# SQLite
# ----------------------------------------------------------------------

db = sqlite3.connect(DATABASE)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Normal image metadata table.
db.execute("""
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    filename    TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    file_size   INTEGER,
    mtime       INTEGER
)
""")

# sqlite-vec table.
#
# ViT-B-32 produces 512-dimensional embeddings.
#
# IMPORTANT:
# This requires sqlite-vec to be loaded into the SQLite connection.
#
# The Python sqlite3 module doesn't automatically know about vec0.
#
# See the note below about loading sqlite-vec.
try:
    db.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS image_embeddings
    USING vec0(
        image_id INTEGER PRIMARY KEY,
        embedding FLOAT[512]
    )
    """)
except sqlite3.OperationalError as e:
    print("Could not create sqlite-vec table:")
    print(e)
    print()
    print("Make sure sqlite-vec is loaded into this SQLite connection.")
    raise

db.commit()


# ----------------------------------------------------------------------
# Find images
# ----------------------------------------------------------------------

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

image_files = [
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print(f"Found {len(image_files)} images.")


# ----------------------------------------------------------------------
# Process images
# ----------------------------------------------------------------------

for index, image_path in enumerate(image_files, 1):

    print(f"[{index}/{len(image_files)}] {image_path}")

    # Avoid indexing the same file twice.
    existing = db.execute(
        "SELECT id FROM images WHERE path = ?",
        (str(image_path),)
    ).fetchone()

    if existing is not None:
        print("  already indexed")
        continue

    try:
        # --------------------------------------------------------------
        # Load image
        # --------------------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # --------------------------------------------------------------
        # Generate CLIP image embedding
        # --------------------------------------------------------------

        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            vector = model.encode_image(image_tensor)

        # Normalize so cosine similarity works nicely.
        vector /= vector.norm(dim=-1, keepdim=True)

        # Move from GPU to CPU.
        vector = vector.cpu().numpy()[0]

        # --------------------------------------------------------------
        # Insert metadata
        # --------------------------------------------------------------

        stat = image_path.stat()

        cursor = db.execute("""
            INSERT INTO images (
                path,
                filename,
                width,
                height,
                file_size,
                mtime
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(image_path),
            image_path.name,
            width,
            height,
            stat.st_size,
            int(stat.st_mtime),
        ))

        image_id = cursor.lastrowid

        # --------------------------------------------------------------
        # Insert vector
        # --------------------------------------------------------------

        # sqlite-vec accepts the vector as a binary float32 blob.
        vector_blob = struct.pack(
            f"{len(vector)}f",
            *vector
        )

        db.execute("""
            INSERT INTO image_embeddings (
                image_id,
                embedding
            )
            VALUES (?, ?)
        """, (
            image_id,
            vector_blob,
        ))

        db.commit()

    except Exception as e:
        print(f"  ERROR: {e}")

        # Don't leave a half-created metadata record.
        db.rollback()


# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------

db.close()

print("Done.")
