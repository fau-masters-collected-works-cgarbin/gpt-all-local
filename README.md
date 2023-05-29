# Using LLMs on private data

This project is a learning exercise on using language models (LLMs) on private data. The goal is to use an LLM to ask questions on private data.

We can divide the project into two parts: ingesting and retrieving data.

1. Ingestion: The goal is to divide data into chunks that fit into the LLM input size. Once the information is divided into chunks, we need to find the most relevant chunks to the question. This is done with a similarity function.
1. Retrieval: Once we determined the most relevant chunks, we can use the LLM to answer the question. To do so, we combine the user question with the relevant chunks and a prompt that instructs the LLM to answer the question.

These two steps are illustrated in the following diagram.

![Overview](./pics/overview.drawio.png)

## Preparing the environment

This is a one-time step. It will create a virtual environment and install the required packages. Skip to the next section if you have already done this.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Ingesting data

## Retrieving data

## Sources

Some projects I learned from.

- [privateGTP](https://github.com/imartinez/privateGPT).

See [this file](./notes.md) for more notes collected during the development of this project.
