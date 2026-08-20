import os
from pathlib import Path

from flask import Flask, request, render_template_string, session
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ainotes-rag-secret")


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".csv",
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".js",
    ".html",
    ".css"
}


def get_api_key():
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured in Render."
        )

    return api_key


def get_loader(filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return PyPDFLoader(filename)

    if extension == ".docx":
        return UnstructuredWordDocumentLoader(filename)

    if extension == ".csv":
        return CSVLoader(filename)

    return TextLoader(
        filename,
        encoding="utf-8"
    )


def build_rag(file_path, original_filename, persist_directory):
    loader = get_loader(file_path)

    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = original_filename

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "The uploaded document contains no readable content."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_api_key()
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="document_assistant",
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=get_api_key(),
        temperature=0.2
    )

    return retriever, llm, len(chunks)


def load_rag(persist_directory):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_api_key()
    )

    vectorstore = Chroma(
        collection_name="document_assistant",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=get_api_key(),
        temperature=0.2
    )

    return retriever, llm


def ask_question(question, retriever, llm):
    results = retriever.invoke(question)

    if not results:
        return (
            "This information is not available "
            "in the uploaded material."
        )

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    prompt = f"""
You are an AI Document Assistant.

Answer the user's question ONLY using
the provided document content.

USER QUESTION:
{question}

DOCUMENT CONTENT:
{context}

RULES:

1. Use only the document content.
2. Do not use outside knowledge.
3. Do not guess.
4. If the document does not contain enough
   information to answer the question, reply exactly:

"This information is not available
in the uploaded material."

5. Give a clear and simple answer.
6. If the question asks for an explanation,
   explain using the document content.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        answer = ""

        for item in content:
            if isinstance(item, dict):
                answer += item.get("text", "")
            else:
                answer += str(item)

        return answer

    return str(content)


HTML = """
<!doctype html>
<html>
<head>
    <title>AI Document Assistant</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 850px;
            margin: 40px auto;
            padding: 20px;
        }

        input, button {
            padding: 10px;
            margin: 8px 0;
            box-sizing: border-box;
        }

        input[type="file"],
        input[type="text"] {
            width: 100%;
        }

        button {
            background: #222;
            color: white;
            border: none;
            cursor: pointer;
            width: 100%;
        }

        .error {
            color: #b00020;
            font-weight: bold;
        }

        .success {
            color: #087f23;
            font-weight: bold;
        }

        .answer {
            white-space: pre-wrap;
            padding: 18px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
</head>

<body>

<h1>🤖 AI Document Assistant</h1>

{% if error %}
<p class="error">{{ error }}</p>
{% endif %}

{% if success %}
<p class="success">{{ success }}</p>
{% endif %}

{% if not uploaded %}

<h3>Upload your document</h3>

<p>
Supported:
PDF, TXT, MD, DOCX, CSV, PY, JAVA,
C, CPP, JS, HTML, CSS
</p>

<form method="post"
      enctype="multipart/form-data">

    <input
        type="file"
        name="document"
        required
    >

    <button type="submit"
            name="action"
            value="upload">
        Upload Document
    </button>

</form>

{% else %}

<p>
<b>Uploaded document:</b>
{{ filename }}
</p>

<p>
Ask questions using only the uploaded document.
</p>

<form method="post">

    <input
        type="text"
        name="question"
        placeholder="Write your question here..."
        required
    >

    <button type="submit"
            name="action"
            value="ask">
        Ask Question
    </button>

</form>

{% if answer %}

<div class="answer">

<b>Question:</b>

{{ question }}

<br><br>

<b>Answer:</b>

{{ answer }}

</div>

{% endif %}

<form method="post">

    <button type="submit"
            name="action"
            value="new">
        Upload Another Document
    </button>

</form>

{% endif %}

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    error = ""
    success = ""
    answer = ""
    question = ""

    uploaded = bool(session.get("document_path"))
    filename = session.get("filename", "")

    if request.method == "POST":

        action = request.form.get("action", "")

        # --------------------------------------
        # UPLOAD DOCUMENT
        # --------------------------------------

        if action == "upload":

            uploaded_file = request.files.get("document")

            if not uploaded_file or not uploaded_file.filename:

                error = (
                    "❌ Please upload a document."
                )

            else:

                original_filename = (
                    uploaded_file.filename
                )

                extension = (
                    Path(original_filename)
                    .suffix
                    .lower()
                )

                if extension not in SUPPORTED_EXTENSIONS:

                    error = (
                        "❌ This type of document is not supported. "
                        "Please upload a supported document."
                    )

                else:

                    try:

                        # Use a persistent temporary directory
                        # for the current Render process.
                        upload_dir = Path("/tmp/ainotes_rag")
                        upload_dir.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        safe_name = secure_filename(
                            original_filename
                        )

                        file_path = upload_dir / safe_name

                        uploaded_file.save(
                            str(file_path)
                        )

                        # Use a unique vector-store directory for
                        # this uploaded document.
                        import uuid
                        rag_id = uuid.uuid4().hex
                        persist_directory = (
                            upload_dir / f"chroma_{rag_id}"
                        )
                        persist_directory.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        retriever, llm, chunk_count = (
                            build_rag(
                                str(file_path),
                                original_filename,
                                str(persist_directory)
                            )
                        )

                        # Store only paths in the session.
                        # The Chroma database itself is stored on disk,
                        # so the next request can load it again.
                        session["document_path"] = str(file_path)
                        session["filename"] = original_filename
                        session["rag_directory"] = str(
                            persist_directory
                        )

                        success = (
                            "✅ Document uploaded successfully. "
                            f"Created {chunk_count} text chunks."
                        )

                        uploaded = True
                        filename = original_filename


                    except Exception as exc:

                        error = (
                            f"❌ Could not read the document: "
                            f"{exc}"
                        )

        # --------------------------------------
        # ASK QUESTION
        # --------------------------------------

        elif action == "ask":

            question = (
                request.form.get("question", "")
                .strip()
            )

            if not question:

                error = (
                    "❌ Please enter a question."
                )

            elif not session.get("document_path"):

                error = (
                    "❌ Please upload a document first."
                )

            else:

                try:

                    rag_directory = session.get(
                        "rag_directory"
                    )

                    if not rag_directory or not Path(
                        rag_directory
                    ).exists():

                        error = (
                            "❌ The uploaded document session "
                            "is no longer available. "
                            "Please upload the document again."
                        )
                        raise RuntimeError(error)

                    # Load the saved Chroma vector database
                    # for every question request.
                    retriever, llm = load_rag(
                        rag_directory
                    )

                    answer = ask_question(
                        question,
                        retriever,
                        llm
                    )

                    uploaded = True
                    filename = session.get(
                        "filename",
                        ""
                    )

                except Exception as exc:

                    error = (
                        f"❌ Error while answering: "
                        f"{exc}"
                    )

        # --------------------------------------
        # NEW DOCUMENT
        # --------------------------------------

        elif action == "new":

            session.pop("document_path", None)
            session.pop("filename", None)
            session.pop("rag_directory", None)

            uploaded = False
            filename = ""
            answer = ""

            success = (
                "Ready for a new document."
            )

    return render_template_string(
        HTML,
        error=error,
        success=success,
        answer=answer,
        question=question,
        uploaded=uploaded,
        filename=filename
    )


@app.route("/health")
def health():
    return "OK", 200


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get("PORT", 10000)
        )
    )
