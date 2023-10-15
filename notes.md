# Using large language models (LLMs) on our own (local) data

## Goal

Use large language models (LLMs) to extract information from our own (local) data.

Typical use case: a company has private documents that were never used to train the model. Therefore, we need to enhance the interaction with the model to get results from these documents.

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
    1. Zero-shot: Potentially larger response time because of vector database lookup and larger prompts.
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

## Vendors, frameworks

### Azure

Azure requires a corporate email to sign up for the service.

- [Main reference](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service)
- [Customizing](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/fine-tuning?pivots=programming-language-studio)
- [Document search](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line)
- [Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)

Technical references:

- [ChatGPT + Enterprise data with Azure OpenAI and Cognitive Search](https://github.com/Azure-Samples/azure-search-openai-demo/)
  - Companion [blog post](https://techcommunity.microsoft.com/t5/ai-applied-ai-blog/revolutionize-your-enterprise-data-with-chatgpt-next-gen-apps-w/ba-p/3762087).
  - [Unofficial implementation](https://github.com/akshata29/chatpdf) (refer to the blog post above).

### Google

- [Vertex AI matching engine](https://cloud.google.com/vertex-ai/docs/matching-engine/overview)

### OpenAI

**NOTE**: OpenAI removed all examples under the `apps` directory ([explanation](https://github.com/openai/openai-cookbook/pull/776)). The links below take you to the repository state before the `apps` directory was removed. Removing them also means they are no longer maintained. As they get older, newer techniques may appear, so use these examples as a starting point and look for newer ones.

- [The "query your data" tutorial](https://github.com/openai/openai-cookbook/tree/9e09df530dbf02c050e4dfff5e4f8e4eb35a26ac/apps/chatbot-kickstarter).
  - See [slides](https://drive.google.com/file/d/1dB-RQhZC_Q1iAsHkNNdkqtxxXqYODFYy/view) first.
  - This is a good conceptual example - a good place to start learning (starting with slides, then code).
- [Enterprise Knowledge Retrieval](https://github.com/openai/openai-cookbook/tree/9e09df530dbf02c050e4dfff5e4f8e4eb35a26ac/apps/enterprise-knowledge-retrieval).
  - _"The notebook is the best place to start, and takes you through an end-to-end workflow for setting up and evaluating a simple back-end knowledge retrieval service"_
  - This is a **very good** notebook to start with. It shows not only the code but also the thought process of putting a good solution in place and evaluating it.
  - It also covers agents, i.e. invoke external tools to complement the LLM.
- [File Q&A](https://github.com/openai/openai-cookbook/tree/9e09df530dbf02c050e4dfff5e4f8e4eb35a26ac/apps/file-q-and-a): _"[U]pload files and ask questions related to their content, and the app will use embeddings and GPT to generate answers from the most relevant files"_.
  - Similar to the "query your data" and "enterprise knowledge retrieval" tutorials, but includes a front-end and backend.
  - Seems to be older than the "knowledge retrieval" tutorial.
- [Using Vector Databases for Embeddings Search](https://github.com/openai/openai-cookbook/blob/main/examples/vector_databases/Using_vector_databases_for_embeddings_search.ipynb).
  - Shows how to use different embedding databases.

### Haystack

- [haystack](https://haystack.deepset.ai/): _"Apply the latest NLP technology to your own data with the use of Haystack's pipeline architecture"._

### LangChain

- [How-To Guides](https://python.langchain.com/en/latest/modules/chains/how_to_guides.html)
  - The ones related to answering questions on documents:
    - [Question Answering](https://github.com/hwchase17/langchain/blob/7047a2c1afce1f1e2e6e4e3e9d94bbf369466a5f/docs/modules/chains/index_examples/question_answering.ipynb): _"how to use LangChain for question answering over a list of documents. It covers four different types of chains: stuff, map_reduce, refine, map_rerank"_.
    - [Retrieval Question/Answering](https://github.com/hwchase17/langchain/blob/7047a2c1afce1f1e2e6e4e3e9d94bbf369466a5f/docs/modules/chains/index_examples/vector_db_qa.ipynb): _"showcases question answering over an index."_

## References

### Projects that run GPT locally

- [privateGPT](https://github.com/imartinez/privateGPT). Based on GPT4All.
  - Shows how to ingest data from different file formats with LangChain.
- [Run ChatGPT-Style Questions Over Your Own Files Using the OpenAI API and LangChain!](https://www.reaminated.com/run-chatgpt-style-questions-over-your-own-files-using-the-openai-api-and-langchain).
- [GPT-4 & LangChain](https://github.com/mayooear/gpt4-pdf-chatbot-langchain): _"Create a ChatGPT Chatbot for Your PDF Files"_.

### Projects that create/publish models

- [GPT4All](https://github.com/nomic-ai/gpt4all)
- [GPT-J](https://www.eleuther.ai/artifacts/gpt-j)
- [MPT-7B](https://www.mosaicml.com/blog/mpt-7b): _"provide a commercially-usable, open-source model that matches (and - in many ways - surpasses) LLaMA-7B."_

### Other projects with LLMs

- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT): Similar to "agents" in LangChain. It may be interesting to see how it is implemented behind the scenes (LangChain is also open source, but it has a lot more than agents -- this one is more focused).

### Vector database and similarity search

- [Faiss: the missing manual](https://www.pinecone.io/learn/faiss/): Covers the basic concepts in the context of Faiss.

### Fine-tuning

- [OpenAI fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)

### Prompt techniques to improve information retrieval

- [OpenAI's Techniques to improve reliability](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md): Illustrates techniques to improve answers, including "Let's think step by step" ([Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) - zero shot) and "chain of thought" ([Language Models Perform Reasoning via Chain of Thought](https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html) - few shot).
- [Unit test writing using a multi-step prompt](https://github.com/openai/openai-cookbook/blob/main/examples/Unit_test_writing_using_a_multi-step_prompt.ipynb): Shows how to use a sequence of prompts to get a response.

### Document parsers/loaders

- [Llama Hub](https://llamahub.ai/)
- [LangChain Document Loaders](https://python.langchain.com/en/latest/modules/indexes/document_loaders.html)
  - Some loaders seem to be based on the ones from Llama Hub.

### Improving search results

- [Question answering using a search API and re-ranking (OpenAI)](https://github.com/openai/openai-cookbook/blob/2ed5c83fb9dc03f5907a4be4e57f128cee4acafc/examples/Question_answering_using_a_search_API.ipynb)
- [Search re-ranking in more details (OpenAI)](https://github.com/openai/openai-cookbook/blob/2ed5c83fb9dc03f5907a4be4e57f128cee4acafc/examples/Search_reranking_with_cross-encoders.ipynb)
- [Hypothetical Document Embeddings: Using hallucinations productively (OpenAI)](https://github.com/openai/openai-cookbook/blob/2ed5c83fb9dc03f5907a4be4e57f128cee4acafc/examples/vector_databases/chroma/hyde-with-chroma-and-openai.ipynb): demonstrates how to use [HyDE](https://arxiv.org/abs/2212.10496) to improve retrieval
- [Fine-tuning a Classifier to Improve Truthfulness (OpenAI)](https://help.openai.com/en/articles/5528730-fine-tuning-a-classifier-to-improve-truthfulness): Discusses how to fine-tune a classifier to remove false results.
- Dynamically chosing [prompts](https://github.com/hwchase17/langchain/blob/7047a2c1afce1f1e2e6e4e3e9d94bbf369466a5f/docs/modules/chains/examples/multi_prompt_router.ipynb) and [retrievers](https://github.com/hwchase17/langchain/blob/7047a2c1afce1f1e2e6e4e3e9d94bbf369466a5f/docs/modules/chains/examples/multi_retrieval_qa_router.ipynb).
- [Moderating the results](https://github.com/hwchase17/langchain/blob/7047a2c1afce1f1e2e6e4e3e9d94bbf369466a5f/docs/modules/chains/examples/moderation.ipynb).

### Other

- [Reddit: The best way to train an LLM on company data (March 2023)](https://www.reddit.com/r/MachineLearning/comments/125qztx/d_the_best_way_to_train_an_llm_on_company_data/): open-ended discussion.
- [psychic](https://github.com/psychic-api/psychic): _"extract and transform unstructured data from SaaS applications like Notion, Slack, Zendesk, Confluence, and Google Drive"_.
- [MLC LLM](https://github.com/mlc-ai/mlc-llm): _"a universal solution that allows any language models to be deployed natively on a diverse set of hardware backends and native applications"_.
- [Prompt auto-evaluator](https://autoevaluator.langchain.com/): Use GPT-4 to evaluate prompts.
- [How write prompts that invoke external tools (agents)](https://github.com/openai/openai-cookbook/blob/59c12ef6dc5ce21ed1f0c83042a70dfeb88084ed/apps/enterprise-knowledge-retrieval/enterprise_knowledge_retrieval.ipynb#L1645): This is the same as the link added above for OpenAI knowledge retrieval. I added it here again to remmeber it also has this interesting piece about agents.
