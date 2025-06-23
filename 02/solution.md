## Question 1

```python
from fastembed import TextEmbedding
```

```python
model_name = 'jinaai/jina-embeddings-v2-small-en'
model = TextEmbedding(model_name)
query = 'I just discovered the course. Can I join now?'
q = next(model.embed([query]))

len(q)
```

    512

```python
q.min()
```

    np.float64(-0.11726373885183883)

**Solution:** -0.11

## Question 2

```python
import numpy as np
```

```python
doc = 'Can I still join the course after the start date?'
embeddings2 = next(model.embed([doc]))
np.dot(q, embeddings2)
```

    np.float64(0.9008528895674548)

**Solution:** 0.9

## Question 3

```python
documents = [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
  'section': 'General course-related questions',
  'question': 'How can we contribute to the course?',
  'course': 'data-engineering-zoomcamp'}]
```

```python
texts = [d['text'] for d in documents]
embeddings = list(model.embed(texts))
sims = np.dot(embeddings, q)
sims
```

    array([0.76296847, 0.81823782, 0.80853974, 0.7133079 , 0.73044992])

```python
sims.argmax()
```

    np.int64(1)

**Solution:** 1

## Question 4

```python
full_texts = [doc['question'] + ' ' + doc['text'] for doc in documents]
```

```python
embeddings_full = list(model.embed(full_texts))
sims_full = np.dot(embeddings_full, q)
sims_full
```

    array([0.85145432, 0.84365942, 0.8408287 , 0.7755158 , 0.80860078])

```python
sims_full.argmax()
```

    np.int64(0)

**Solution:** 0

## Question 5

```python
import pandas as pd
pd.DataFrame(TextEmbedding.list_supported_models())["dim"].min()
```

    np.int64(384)

**Solution:** 384

## Question 6

```python
import requests

model_q6 = TextEmbedding(model_name='BAAI/bge-small-en')

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_raw = requests.get(docs_url).json()

docs = []
for course in docs_raw:
    if course['course'] == 'machine-learning-zoomcamp':
        for doc in course['documents']:
            docs.append(doc['question'] + ' ' + doc['text'])

len(docs)
```

    375

```python
doc_embeddings = model_q6.embed(docs)
q_embed_q6 = next(model_q6.embed([query]))
```

```python
scores = []
for de in doc_embeddings:
    score = de.dot(q_embed_q6)
    scores.append(score)
```

```python
np.max(scores)
```

    np.float32(0.8703172)

**Solution:** 0.87
