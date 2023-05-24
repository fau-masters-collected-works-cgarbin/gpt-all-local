# Using large language models (LLMs) on our own (local) data

## Goal

Use large language models (LLMs) to extract information from our own (local) data.

Typical use case: a company has private documents that where never used to train the model. Therefore we need to enhance the interaction with the model to get results from these documents.

Possible solutions:

1. Fine-tuning: fine-tune a pre-trained model on our own data.
   1. Train the model on our own data.
1. Zero-shot learning: use a pre-trained model without fine-tuning it on our own data.
   1. Extract information with a vector (embedding) of our own data.
   1. Feed that vector to the pre-trained model as part of the prompt.

## What to consider when selecting a solution

Items to consider when selecting a solution:

1. License
   1. Commercial vs. non-commercial use.
   1. GPLv3 vs. others.
1. Time to make the solution available.
   1. Fine-tuning: time to train the model.
   1. Zero-shot learning: time to create a vector database that will be used to extract information from our own data.
1. Time to update the solution.
   1. Fine-tuning: time to re-train the model on changed data.
   1. Zero-shot learning: time to update the vector database.
1. Cost
    1. Fine-tuning: cost of training the model.
    1. Zero-shot learning: cost of creating the vector database and (potential) costs of larger prompts (pay per token).
1. Accuracy
    1. Fine-tuning: potentially less accurate than zero-shot learning because the trained model is probabilistic, not deterministic.
    1. Zero-shot learning: accuracy of the model on our own data.
1. Response time
    1. Fine-tuning: potentially faster response time because of smaller prompts.
    1. Zero-shot Potentially larger response time because of vector database lookup and larger prompts.
1. Data privacy
    1. Fine-tuning: data privacy is a concern because the model is trained on our own data.
    1. Zero-shot learning: data privacy is less of a concern because the model is not trained on our own data (but we still expose the prompt to the model).
1. Security
   1. How to filter out private documents from the model?
   1. How to protect individuals' privacy?
   1. What regulations apply, e.g. GDPR, CCPA, HIPAA, etc.?
   1. What audit and compliance requirements apply?
   1. Resistance to poisoning attacks.
   1. Resistance to backdoor attacks.
   1. Resistance to model inversion attacks (extract data from the model).
   1. Resistance to membership inference attacks (determine if a data point was used to train the model).
   1. Resistance to offensive language attacks (generate offensive language).
   1. Resistance to model stealing attacks (steal the model).

## Investigations

### Azure

Azure requires a corporate email to sign up for the service.

- [Main reference](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service)
- [Customizing](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/fine-tuning?pivots=programming-language-studio)
- [Document search](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line)
- [Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)

### OpenAI

### LangChain

### Other

- [GPT4All](https://github.com/nomic-ai/gpt4all)
- [privateGPT](https://github.com/imartinez/privateGPT). Based on GPT4All.
- [GPT-J](https://www.eleuther.ai/artifacts/gpt-j)
- [MPT-7B](https://www.mosaicml.com/blog/mpt-7b)

## References

### Vector database and similarity search

- [Faiss: the missing manual](https://www.pinecone.io/learn/faiss/): Covers the basic concepts in the context of Faiss.

### RLHF

- TODO: add references

### Fine tuning

- [OpenAI fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)

### Zero-shot learning

TODO: add references

### General

- [Reddit: The best way to train an LLM on company data (March 2023)](https://www.reddit.com/r/MachineLearning/comments/125qztx/d_the_best_way_to_train_an_llm_on_company_data/): open-ended discussion.
- [psychic](https://github.com/psychic-api/psychic): _"extract and transform unstructured data from SaaS applications like Notion, Slack, Zendesk, Confluence, and Google Drive"_.
- [MLC LLM](https://github.com/mlc-ai/mlc-llm): _"a universal solution that allows any language models to be deployed natively on a diverse set of hardware backends and native applications"_.
