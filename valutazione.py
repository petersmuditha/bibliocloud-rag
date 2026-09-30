from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from rag import create_qa_chain

# 1. Configure Groq as evaluator
evaluator_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    timeout=None,
)

# 2. Local embeddings
evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Wrap for RAGAS
ragas_llm = LangchainLLMWrapper(evaluator_llm)
ragas_emb = LangchainEmbeddingsWrapper(evaluator_embeddings)

# 4. Define metrics
m1 = Faithfulness(llm=ragas_llm)
m2 = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
m1.n = 1
m2.n = 1

# 5. Evaluation dataset
eval_data = {
    "question": [
        "How many books can a student borrow?",
        "What is the fine for late returns?",
        "What are the opening hours on Saturday?",
        "Can I renew a reserved book?",
        "What is the phone number of the library?"
    ],
    "answer": [],
    "contexts": [],
    "ground_truth": [
        "Maximum 5 books at a time",
        "1 euro per day per book",
        "Saturday: 9:00 to 18:00",
        "No, reserved books cannot be renewed",
        "Information not found in the knowledge base"
    ]
}

# 6. Generate answers and contexts
print("Creating QA chain...")
qa_chain = create_qa_chain()

print("Generating answers and contexts...")
for question in eval_data["question"]:
    print(f"  Processing: {question[:50]}...")
    result = qa_chain.invoke({"query": question})
    eval_data["answer"].append(result["result"])
    eval_data["contexts"].append([doc.page_content for doc in result["source_documents"]])

# 7. Convert to Dataset
dataset = Dataset.from_dict(eval_data)

# 8. RunConfig
config = RunConfig(
    timeout=120,
    max_retries=5,
    max_workers=1
)

# 9. Evaluate
print("\nStarting evaluation with Groq...")
results = evaluate(
    dataset,
    metrics=[m1, m2],
    run_config=config
)

# 10. Print results
print("\n=== EVALUATION RESULTS ===")
print(f"Faithfulness: {results['faithfulness']}")
print(f"Answer Relevancy: {results['answer_relevancy']}")

# 11. Per-question details
print("\n=== PER-QUESTION DETAILS ===")
df = results.to_pandas()
for i, row in df.iterrows():
    print(f"\nQ{i+1}: {row['question'][:60]}...")
    print(f"  Answer: {row['answer'][:80]}...")
    print(f"  Faithfulness: {row['faithfulness']}")
    print(f"  Answer Relevancy: {row['answer_relevancy']}")