# Domain-Glossary

## Environment

Python 3.6.8

## Steps

### Step 1: Install libraries

`pip install -r requirements.txt`

### Step 2: Prepare Glove word vectors

1. Download Glove from http://nlp.stanford.edu/data/glove.840B.300d.zip

2. Unzip the zip file

3. copy glove.840B.300d.txt to resources/glove/glove.txt

### Step 3: Prepare corpora

1. Prepare general technical corpus (HTML files) for computing `Generality(term)` (e.g., JDK and Python documentation)

2. Prepare domain-specific corpus (HTML and source code files) from different projects (e.g., DL4J, tensorflow and pytorch for deep learning domain)

3. Convert the data into the desired format:

There are three attributes for each HTML entry：

- type: "javadoc" or "html"
- source: project name (e.g., "DL4J" or "pytorch")
- content: HTML text

```
An example of HTML data format:
[
  ("javadoc", "DL4J", content),
  ("html", "pytorch", content),
  ...
]
```

There are three attributes for each code entry：

- type: "java" or "python"
- path: source code file path (note: start with the root (or its children) directory of a specific project, e.g., "torch/nn/.." instead of "xxx/data/code/torch/..")
- content: code text

```
An example of code data format:
[
  ("java", "deeplearning4j/deeplearning4j-nn/.../LSTM.java", content),
  ("python", "torch/nn/../rnn.py", content),
  ...
]
```

## Step 4: Build general tecnnical corpus

```
cd Domain-Glossary
export PYTHONPATH=.
python script/build_general_corpus.py
```

## Step 5: Run pipeline

`python script/app.py`
