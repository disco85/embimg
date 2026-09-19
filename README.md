# embimg
Images embedding: LLM usage for search by description. It is very simple
Proof of Concept - all code is written by AI. Following the steps below you
can create small sqlite3 DB with verctor representation of "knowledge" about
PNG, JPEG, WEP etc images and then to search for image using free text
description.

# Fixture images

All images are in `./images`, they are super simple:

```
g1.png               abstract geometry
mam2.jpeg            mammoths
ufo-finnish1.jpeg    UFO
ufo-finnish9.jpeg    UFO
```

# Installation

```bash
$ python -m venv .venv
$ .venv/bin/pip install -r requirements.txt
```

It will download about 5-6Gb libraries into the virtual environment
`.venv/`.

# Creating of DB

The next step is to create empty DB - it is sqlite3:

```bash
sqlite3 images.sqlite < schema.sql
```

# Create the index of images

Now you can indexate all images from `./images` directory:

```bash
$ .venv/bin/python indexate.py
```

Wait why it will do whole the job including download of models (about
600Mb).

# Query

Now you can query DB:

```bash
$ .venv/bin/python query.py 
Loading model...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Enter query: animals like elephants
Query: animals like elephants
1.186974	images/mam2.jpeg
1.318390	images/ufo-finnish1.jpeg
1.341814	images/ufo-finnish9.jpeg
1.380846	images/g1.png
```

So, the query was "animals like elephants" and the image with the smallest
distance to this description is `images/mam2.jpeg` which is correct. Other
queries:

```bash
$ .venv/bin/python query.py 
Loading model...
Enter query: ufo
Query: ufo
1.157540	images/ufo-finnish1.jpeg
1.170334	images/ufo-finnish9.jpeg
1.291521	images/g1.png
1.326028	images/mam2.jpeg
```

Correct, now distances 1.15, 1.17 point to ufo images. And:

```bash
$ .venv/bin/python query.py 
Loading model...
Enter query: sierpinski  
Query: sierpinski
1.193816	images/g1.png
1.272899	images/ufo-finnish9.jpeg
1.303475	images/ufo-finnish1.jpeg
1.345176	images/mam2.jpeg
```