
# Document Q&A Chatbot — RAG Pipeline

A retrieval-augmented generation (RAG) application that lets you upload any PDF and ask natural language questions about it. The system retrieves the most relevant passages from the document and uses an LLM to generate grounded, accurate answers — without hallucinating content that isn't there.

---

## What the project does

Most LLMs can only answer questions based on what they were trained on. This project solves a practical problem: how do you ask questions about a document the model has never seen?

The answer is RAG. When you upload a PDF, the app splits it into overlapping chunks, converts each chunk into a semantic vector using OpenAI embeddings, and stores those vectors locally in ChromaDB. When you ask a question, the same embedding model converts your question into a vector, finds the 4 most similar chunks via cosine similarity search, and passes those chunks as context to GPT-4o-mini, which generates an answer grounded strictly in the retrieved text. The app also surfaces the source chunks used so answers are fully traceable.

---

## Tech stack

| Tool | Role |
|------|------|
| LangChain | Orchestration layer — connects document loading, chunking, embedding, retrieval, and LLM into a single pipeline |
| OpenAI API (`text-embedding-3-small`) | Converts text chunks and user queries into semantic vectors |
| OpenAI API (`gpt-4o-mini`) | Generates natural language answers grounded in retrieved context |
| ChromaDB | Local vector database — stores and queries document embeddings |
| Streamlit | Frontend UI — PDF upload, question input, answer display, source chunk viewer |
| PyPDF | Extracts raw text from uploaded PDF files |
| python-dotenv | Loads API keys securely from a `.env` file |

---

## How to run locally

**1. Clone the repository**
```bash
git clone https://github.com/aadhyanagar08/rag-chatbot.git
cd rag-chatbot
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install langchain langchain-community langchain-openai openai chromadb streamlit pypdf python-dotenv tiktoken
```

**4. Add your OpenAI API key**

Create a `.env` file in the root of the project:
```
OPENAI_API_KEY=sk-your-key-here
```

Get your key at [platform.openai.com](https://platform.openai.com). Free tier credits are sufficient for testing.

**5. Run the app**
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`. Upload a PDF and start asking questions.

---

## Evaluation results

To assess output quality, the chatbot was tested against a university academic calendar PDF across 15 questions spanning basic comprehension, specific fact retrieval, structural rule extraction, cross-document reasoning, and edge case handling. Each answer was manually reviewed against the source document and scored as correct (grounded in retrieved chunks) or incorrect (hallucinated or not found).

### Results by category

| # | Category | Question | Result |
|---|----------|----------|--------|
| 1 | Basic understanding | What are the main types of degrees covered in the document? |  Correct |
| 2 | Basic understanding | Summarize the graduation requirements for an Honours Bachelor Degree. |  Correct |
| 3 | Basic understanding | What is the difference between Honours Bachelor and Four-Year Bachelor requirements? |  Not retrieved |
| 4 | Specific retrieval | How many total courses are required to graduate with an Honours Bachelor Degree? |  Correct |
| 5 | Specific retrieval | What is the minimum overall average required for an Honours Bachelor Degree? |  Correct |
| 6 | Specific retrieval | How many senior courses are required for a Four-Year Bachelor Degree? |  Correct |
| 7 | Specific retrieval | What is the minimum mark required in each course for graduation? |  Correct |
| 8 | Structure + rules | What are the first-year course requirements? |  Correct |
| 9 | Structure + rules | What are the essay course requirements and are there any exceptions? |  Correct |
| 10 | Structure + rules | What are the residency requirements for Honours Bachelor students? |  Not retrieved |
| 11 | Reasoning / comparison | Compare the senior course requirements between Three-Year and Four-Year degrees. |  Correct |
| 12 | Reasoning / comparison | How do module requirements differ between Honours and Three-Year degrees? |  Correct |
| 13 | Reasoning / comparison | What is the difference in average requirements between Honours and Four-Year degrees? |  Not retrieved |
| 14 | Edge cases | What happens if a student in an Honours program has an average of 68% in their specialization? |  Correct |
| 15 | Edge cases | What exceptions apply to students admitted with advanced standing credits? |  Correct |

### Summary

| Metric | Score |
|--------|-------|
| Total questions | 15 |
| Correctly answered | 12 |
| Not retrieved | 3 |
| Hallucinations | 0 |
| Faithfulness score | **80%** |

### Analysis

The pipeline achieved **80% faithfulness with zero hallucinations** — every answer was either grounded in the retrieved source chunks or the model correctly identified that the information was not available in the context. This is the intended behaviour of a well-prompted RAG system: the model never fabricated an answer.

The 3 missed retrievals share a common pattern: they required the model to synthesize information spread across multiple non-adjacent sections of the document (comparison questions and residency rules that appear in separate policy clauses). This is a known limitation of chunk-level retrieval — when the answer requires joining context from multiple distant passages, a single top-k retrieval pass may not surface all relevant chunks.

**Potential improvements:** increasing `k` from 4 to 6 for comparison-type queries, implementing a re-ranking step (e.g. Cohere Rerank), or using a hybrid search approach combining keyword and semantic retrieval. These are standard next steps in production RAG systems.