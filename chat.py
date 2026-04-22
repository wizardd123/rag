import warnings
warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CHROMA_PATH = "db"

PROMPT_TEMPLATE = """
You are a knowledgeable history assistant.

Answer the question using ONLY the context below.

Rules:
- Write a detailed answer (6-10 sentences)
- Write in one paragraph
- Combine information from different parts of the context
- Explain clearly

Context:
{context}

Question: {question}

Answer:
"""


def main():
    # ✅ Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # ✅ Vector DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # ✅ Load FLAN-T5 (NO pipeline → NO error)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    print("🤖 RAG Chatbot Ready! (type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        # 🔍 Retrieve best match
        results = db.similarity_search(query, k=4)

        if not results:
            print("\n🤖 Answer:\n No relevant data found.\n")
            continue

        # 🧹 Clean context
        context = " ".join([doc.page_content for doc in results])
        context = context.replace("\n", " ")

        # 🧠 Prompt
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        final_prompt = prompt.format(context=context, question=query)

        # 🤖 Generate answer (correct way)
        inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )


        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 🧼 Clean output
        answer = answer.replace("\n", " ").strip()

        print("\n🤖 Answer:\n", answer, "\n")


if __name__ == "__main__":
    main()
