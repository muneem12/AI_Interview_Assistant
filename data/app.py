import streamlit as st
import re
from datetime import datetime

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Interview Preparation Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Interview Preparation Assistant")

st.write(
    "Prepare for AI Engineer interviews using your resume, "
    "interview topics, RAG, mock interviews and AI feedback."
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "questions": [],
    "current_question": 0,
    "evaluations": [],
    "answer_submitted": False,
    "resume_text": "",
    "resume_vectorstore": None,
    "guide_vectorstore": None,
    "chat_history": [],
    "interview_started": False,
    "selected_topic": "Python"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# INTERVIEW GUIDE
# ============================================================

GUIDE_FILE = "interview_guide.txt"


def load_interview_guide():

    try:

        with open(
            GUIDE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except FileNotFoundError:

        return """
AI Engineer Interview Guide

Python:
Important topics include variables, functions, OOP, lists,
dictionaries, exception handling, and file handling.

Machine Learning:
Important topics include supervised learning, unsupervised learning,
regression, classification, overfitting, underfitting,
and evaluation metrics.

Deep Learning:
Important topics include neural networks, activation functions,
backpropagation, CNNs, RNNs, LSTMs, and Transformers.

NLP:
Important topics include tokenization, embeddings, attention,
sequence models, and text classification.

Generative AI:
Important topics include LLMs, prompting, fine-tuning, RAG,
embeddings, vector databases, and agents.

LangChain:
Important topics include prompt templates, chains, output parsers,
document loaders, text splitters, retrievers, and vector stores.
"""


guide_text = load_interview_guide()


# ============================================================
# TOPIC DATA
# ============================================================

TOPIC_DATA = {

    "Python": {

        "topics": [
            "Variables",
            "Functions",
            "OOP",
            "Lists",
            "Dictionaries",
            "Exception handling",
            "File handling"
        ],

        "questions": [

            {
                "question": "What is a variable in Python?",
                "keywords": [
                    "variable",
                    "value",
                    "store",
                    "data"
                ]
            },

            {
                "question": "What is a function in Python?",
                "keywords": [
                    "function",
                    "def",
                    "parameter",
                    "argument",
                    "return"
                ]
            },

            {
                "question": "What is OOP in Python?",
                "keywords": [
                    "object",
                    "class",
                    "method",
                    "attribute"
                ]
            },

            {
                "question": "What is a list in Python?",
                "keywords": [
                    "list",
                    "ordered",
                    "index",
                    "append",
                    "remove"
                ]
            },

            {
                "question": "What is a dictionary in Python?",
                "keywords": [
                    "dictionary",
                    "key",
                    "value",
                    "key-value"
                ]
            },

            {
                "question": "What is exception handling in Python?",
                "keywords": [
                    "try",
                    "except",
                    "exception",
                    "error",
                    "finally"
                ]
            },

            {
                "question": "How do you handle files in Python?",
                "keywords": [
                    "file",
                    "open",
                    "read",
                    "write",
                    "close"
                ]
            }
        ]
    },


    "Machine Learning": {

        "topics": [
            "Supervised learning",
            "Unsupervised learning",
            "Regression",
            "Classification",
            "Overfitting",
            "Underfitting",
            "Evaluation metrics"
        ],

        "questions": [

            {
                "question": "What is supervised learning?",
                "keywords": [
                    "labeled",
                    "label",
                    "input",
                    "output",
                    "training"
                ]
            },

            {
                "question": "What is unsupervised learning?",
                "keywords": [
                    "unlabeled",
                    "clustering",
                    "patterns",
                    "groups"
                ]
            },

            {
                "question": "What is regression in machine learning?",
                "keywords": [
                    "regression",
                    "continuous",
                    "value",
                    "prediction"
                ]
            },

            {
                "question": "What is classification in machine learning?",
                "keywords": [
                    "classification",
                    "class",
                    "category",
                    "label"
                ]
            },

            {
                "question": "What is overfitting?",
                "keywords": [
                    "overfitting",
                    "training",
                    "test",
                    "generalize"
                ]
            },

            {
                "question": "What is underfitting?",
                "keywords": [
                    "underfitting",
                    "simple",
                    "model",
                    "training"
                ]
            },

            {
                "question": "What are evaluation metrics in machine learning?",
                "keywords": [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "metric"
                ]
            }
        ]
    },


    "Deep Learning": {

        "topics": [
            "Neural networks",
            "Activation functions",
            "Backpropagation",
            "CNNs",
            "RNNs",
            "LSTMs",
            "Transformers"
        ],

        "questions": [

            {
                "question": "What is a neural network?",
                "keywords": [
                    "neural",
                    "network",
                    "neuron",
                    "layer",
                    "weights"
                ]
            },

            {
                "question": "What is an activation function?",
                "keywords": [
                    "activation",
                    "relu",
                    "sigmoid",
                    "function"
                ]
            },

            {
                "question": "What is backpropagation?",
                "keywords": [
                    "backpropagation",
                    "gradient",
                    "error",
                    "weights"
                ]
            },

            {
                "question": "What is a CNN?",
                "keywords": [
                    "cnn",
                    "convolution",
                    "image",
                    "filter",
                    "pooling"
                ]
            },

            {
                "question": "What is an RNN?",
                "keywords": [
                    "rnn",
                    "sequence",
                    "hidden",
                    "time"
                ]
            },

            {
                "question": "What is an LSTM?",
                "keywords": [
                    "lstm",
                    "memory",
                    "cell",
                    "sequence"
                ]
            },

            {
                "question": "What is a Transformer?",
                "keywords": [
                    "transformer",
                    "attention",
                    "self-attention",
                    "sequence"
                ]
            }
        ]
    },


    "NLP": {

        "topics": [
            "Tokenization",
            "Embeddings",
            "Attention",
            "Sequence models",
            "Text classification"
        ],

        "questions": [

            {
                "question": "What is tokenization in NLP?",
                "keywords": [
                    "token",
                    "tokenization",
                    "words",
                    "text"
                ]
            },

            {
                "question": "What are embeddings in NLP?",
                "keywords": [
                    "embedding",
                    "vector",
                    "representation",
                    "word"
                ]
            },

            {
                "question": "What is attention in NLP?",
                "keywords": [
                    "attention",
                    "focus",
                    "important",
                    "words"
                ]
            },

            {
                "question": "What are sequence models?",
                "keywords": [
                    "sequence",
                    "rnn",
                    "lstm",
                    "gru"
                ]
            },

            {
                "question": "What is text classification?",
                "keywords": [
                    "text",
                    "classification",
                    "category",
                    "label"
                ]
            }
        ]
    },


    "Generative AI": {

        "topics": [
            "LLMs",
            "Prompting",
            "Fine-tuning",
            "RAG",
            "Embeddings",
            "Vector databases",
            "Agents"
        ],

        "questions": [

            {
                "question": "What is an LLM?",
                "keywords": [
                    "llm",
                    "language",
                    "model",
                    "text"
                ]
            },

            {
                "question": "What is prompting?",
                "keywords": [
                    "prompt",
                    "instruction",
                    "input",
                    "model"
                ]
            },

            {
                "question": "What is fine-tuning?",
                "keywords": [
                    "fine-tuning",
                    "training",
                    "dataset",
                    "model"
                ]
            },

            {
                "question": "What is RAG?",
                "keywords": [
                    "rag",
                    "retrieval",
                    "generation",
                    "documents",
                    "context"
                ]
            },

            {
                "question": "What are embeddings?",
                "keywords": [
                    "embedding",
                    "vector",
                    "representation"
                ]
            },

            {
                "question": "What is a vector database?",
                "keywords": [
                    "vector",
                    "database",
                    "similarity",
                    "search"
                ]
            },

            {
                "question": "What is an AI agent?",
                "keywords": [
                    "agent",
                    "tool",
                    "action",
                    "reasoning"
                ]
            }
        ]
    },


    "LangChain": {

        "topics": [
            "Prompt templates",
            "Chains",
            "Output parsers",
            "Document loaders",
            "Text splitters",
            "Retrievers",
            "Vector stores"
        ],

        "questions": [

            {
                "question": "What are prompt templates in LangChain?",
                "keywords": [
                    "prompt",
                    "template",
                    "variables"
                ]
            },

            {
                "question": "What is a chain in LangChain?",
                "keywords": [
                    "chain",
                    "sequence",
                    "components"
                ]
            },

            {
                "question": "What are output parsers?",
                "keywords": [
                    "output",
                    "parser",
                    "format",
                    "structured"
                ]
            },

            {
                "question": "What are document loaders?",
                "keywords": [
                    "document",
                    "loader",
                    "pdf",
                    "load"
                ]
            },

            {
                "question": "What are text splitters?",
                "keywords": [
                    "split",
                    "chunk",
                    "text",
                    "document"
                ]
            },

            {
                "question": "What is a retriever?",
                "keywords": [
                    "retriever",
                    "retrieve",
                    "documents",
                    "search"
                ]
            },

            {
                "question": "What is a vector store?",
                "keywords": [
                    "vector",
                    "store",
                    "embedding",
                    "similarity"
                ]
            }
        ]
    }
}


# ============================================================
# LOAD QWEN
# ============================================================

@st.cache_resource
def load_model():

    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=150,
        device=-1,
        return_full_text=False
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return ChatHuggingFace(
        llm=llm
    )


@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


chat_model = load_model()
embeddings = load_embeddings()


# ============================================================
# BUILD GUIDE RAG
# ============================================================

@st.cache_resource
def build_guide_vectorstore():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    from langchain_core.documents import Document

    document = Document(
        page_content=guide_text,
        metadata={"source": "Interview Guide"}
    )

    chunks = splitter.split_documents(
        [document]
    )

    return FAISS.from_documents(
        chunks,
        embeddings
    )


if st.session_state.guide_vectorstore is None:

    st.session_state.guide_vectorstore = (
        build_guide_vectorstore()
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Interview Setup")

    topic = st.selectbox(
        "Select Topic",
        list(TOPIC_DATA.keys())
    )

    st.session_state.selected_topic = topic

    st.divider()

    st.write("### Project Features")

    st.write("✅ Resume Analysis")
    st.write("✅ Skill Gap Analysis")
    st.write("✅ RAG")
    st.write("✅ Mock Interview")
    st.write("✅ AI Feedback")
    st.write("✅ Final Report")

    st.divider()

    st.caption(
        "AI Interview Preparation Assistant"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📄 Resume Analysis",
        "🎯 Skill Gap",
        "🎤 Mock Interview",
        "💬 AI Assistant"
    ]
)


# ============================================================
# TAB 1 — RESUME ANALYSIS
# ============================================================

with tab1:

    st.header("📄 Resume Analysis")

    uploaded_resume = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"]
    )

    if uploaded_resume:

        temp_path = "uploaded_resume.pdf"

        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(
                uploaded_resume.getbuffer()
            )

        try:

            loader = PyPDFLoader(
                temp_path
            )

            pages = loader.load()

            resume_text = "\n".join(
                page.page_content
                for page in pages
            )

            st.session_state.resume_text = resume_text

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=80
            )

            chunks = splitter.split_documents(
                pages
            )

            st.session_state.resume_vectorstore = (
                FAISS.from_documents(
                    chunks,
                    embeddings
                )
            )

            st.success(
                "Resume uploaded successfully!"
            )

            st.subheader("Extracted Resume")

            st.text_area(
                "Resume Content",
                resume_text,
                height=300
            )

            st.download_button(
                "Download Extracted Resume Text",
                resume_text,
                file_name="extracted_resume.txt"
            )

        except Exception as e:

            st.error(
                f"Could not process resume: {e}"
            )


# ============================================================
# TAB 2 — SKILL GAP
# ============================================================

with tab2:

    st.header("🎯 Resume vs Interview Guide")

    if not st.session_state.resume_text:

        st.info(
            "Upload your resume in the Resume Analysis tab first."
        )

    else:

        resume_lower = (
            st.session_state.resume_text.lower()
        )

        topics = TOPIC_DATA[topic]["topics"]

        present_topics = []
        missing_topics = []

        for item in topics:

            words = item.lower().split()

            found = False

            for word in words:

                if len(word) > 3 and word in resume_lower:

                    found = True
                    break

            if found:

                present_topics.append(item)

            else:

                missing_topics.append(item)


        col1, col2 = st.columns(2)

        with col1:

            st.subheader("✅ Skills Already Present")

            if present_topics:

                for item in present_topics:

                    st.write(
                        f"✅ {item}"
                    )

            else:

                st.write(
                    "No clearly matching topics found."
                )


        with col2:

            st.subheader("⚠️ Topics to Improve")

            if missing_topics:

                for item in missing_topics:

                    st.write(
                        f"⚠️ {item}"
                    )

            else:

                st.write(
                    "No obvious gaps found."
                )


        st.divider()

        st.subheader(
            "📚 Preparation Recommendation"
        )

        if missing_topics:

            st.write(
                "Focus your preparation on:"
            )

            for item in missing_topics:

                st.write(
                    f"• {item}"
                )

        else:

            st.success(
                "Your resume covers the main interview topics."
            )


# ============================================================
# TAB 3 — MOCK INTERVIEW
# ============================================================

with tab3:

    st.header("🎤 Mock Interview")

    st.write(
        f"Selected Topic: **{topic}**"
    )

    # --------------------------------------------------------
    # START INTERVIEW
    # --------------------------------------------------------

    if not st.session_state.interview_started:

        st.write(
            "You will answer 5 interview questions."
        )

        if st.button(
            "🚀 Start Interview",
            type="primary"
        ):

            # Select 5 questions
            available_questions = (
                TOPIC_DATA[topic]["questions"]
            )

            st.session_state.questions = (
                available_questions[:5]
            )

            st.session_state.current_question = 0
            st.session_state.evaluations = []
            st.session_state.answer_submitted = False
            st.session_state.interview_started = True

            st.rerun()


    # --------------------------------------------------------
    # INTERVIEW IN PROGRESS
    # --------------------------------------------------------

    if (
        st.session_state.interview_started
        and
        st.session_state.current_question < 5
    ):

        current_index = (
            st.session_state.current_question
        )

        current_question = (
            st.session_state.questions[current_index]
        )

        question_text = current_question["question"]

        keywords = current_question["keywords"]


        # Progress
        progress = (
            current_index / 5
        )

        st.progress(
            progress,
            text=f"Question {current_index + 1} of 5"
        )


        st.subheader(
            f"Question {current_index + 1}"
        )

        st.write(
            f"### {question_text}"
        )


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        answer = st.text_area(
            "Your Answer",
            height=180,
            key=f"answer_{current_index}"
        )


        # ----------------------------------------------------
        # SUBMIT
        # ----------------------------------------------------

        if not st.session_state.answer_submitted:

            if st.button(
                "Submit Answer",
                type="primary"
            ):

                answer_clean = answer.strip()

                # --------------------------------------------
                # SCORE
                # --------------------------------------------

                if not answer_clean:

                    score = 0

                else:

                    answer_lower = (
                        answer_clean.lower()
                    )

                    matched = []

                    for keyword in keywords:

                        if keyword.lower() in answer_lower:

                            matched.append(keyword)


                    # Remove duplicates
                    matched = list(
                        set(matched)
                    )

                    match_count = len(matched)

                    # More realistic scoring
                    if len(answer_clean) < 15:

                        score = 1

                    elif match_count == 0:

                        score = 2

                    elif match_count == 1:

                        score = 4

                    elif match_count == 2:

                        score = 6

                    elif match_count == 3:

                        score = 8

                    else:

                        score = 10


                # --------------------------------------------
                # AI FEEDBACK
                # --------------------------------------------

                feedback_prompt = f"""
You are an interview evaluator.

Question:
{question_text}

Candidate Answer:
{answer_clean}

Give exactly:

Strength: one short sentence.

Improvement: one short sentence.

Do not give a score.
Do not invent information.
"""

                try:

                    feedback_response = (
                        chat_model.invoke(
                            feedback_prompt
                        )
                    )

                    feedback = (
                        feedback_response.content
                    )

                except Exception:

                    feedback = (
                        "Strength: You attempted the question.\n\n"
                        "Improvement: Explain the concept more clearly "
                        "and include an example where appropriate."
                    )


                # --------------------------------------------
                # SAVE EVALUATION
                # --------------------------------------------

                st.session_state.evaluations.append(
                    {
                        "question": question_text,
                        "answer": answer_clean,
                        "score": score,
                        "feedback": feedback
                    }
                )

                st.session_state.answer_submitted = True

                st.rerun()


        # ----------------------------------------------------
        # SHOW EVALUATION
        # ----------------------------------------------------

        if st.session_state.answer_submitted:

            result = (
                st.session_state.evaluations[-1]
            )

            st.divider()

            st.subheader(
                "📊 Evaluation"
            )

            st.metric(
                "Score",
                f"{result['score']}/10"
            )

            st.write(
                result["feedback"]
            )


            # -----------------------------------------------
            # NEXT QUESTION
            # -----------------------------------------------

            if current_index < 4:

                if st.button(
                    "➡️ Next Question",
                    type="primary"
                ):

                    st.session_state.current_question += 1

                    st.session_state.answer_submitted = False

                    st.rerun()

            else:

                st.success(
                    "🎉 All 5 questions completed!"
                )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if (
        st.session_state.interview_started
        and
        len(st.session_state.evaluations) == 5
        and
        st.session_state.current_question == 4
        and
        st.session_state.answer_submitted
    ):

        st.divider()

        st.header(
            "🏆 Final Interview Result"
        )

        scores = [
            item["score"]
            for item in st.session_state.evaluations
        ]

        average_score = (
            sum(scores) / len(scores)
        )

        st.metric(
            "Overall Score",
            f"{average_score:.1f}/10"
        )


        # -----------------------------------------------
        # PERFORMANCE MESSAGE
        # -----------------------------------------------

        if average_score >= 8:

            st.success(
                "Excellent interview performance! 🚀"
            )

        elif average_score >= 6:

            st.info(
                "Good performance. Keep practicing. 👍"
            )

        elif average_score >= 4:

            st.warning(
                "Average performance. Focus on your weak topics. 📚"
            )

        else:

            st.error(
                "You need more preparation before the interview. 💪"
            )


        # -----------------------------------------------
        # SCORE BREAKDOWN
        # -----------------------------------------------

        st.subheader(
            "Question-wise Scores"
        )

        for i, item in enumerate(
            st.session_state.evaluations
        ):

            st.write(
                f"**Q{i + 1}: {item['score']}/10**"
            )

            st.write(
                item["question"]
            )


        # -----------------------------------------------
        # WEAK QUESTIONS
        # -----------------------------------------------

        weak_questions = [
            item
            for item in st.session_state.evaluations
            if item["score"] <= 4
        ]

        if weak_questions:

            st.subheader(
                "⚠️ Questions to Improve"
            )

            for item in weak_questions:

                st.write(
                    f"• {item['question']} — "
                    f"{item['score']}/10"
                )


        # -----------------------------------------------
        # REPORT GENERATION
        # -----------------------------------------------

        report = []

        report.append(
            "AI INTERVIEW PREPARATION ASSISTANT"
        )

        report.append(
            "=" * 45
        )

        report.append(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        report.append(
            f"Topic: {topic}"
        )

        report.append(
            f"Overall Score: {average_score:.1f}/10"
        )

        report.append("")

        report.append(
            "QUESTION-WISE RESULTS"
        )

        report.append(
            "-" * 30
        )


        for i, item in enumerate(
            st.session_state.evaluations
        ):

            report.append(
                f"\nQuestion {i + 1}:"
            )

            report.append(
                item["question"]
            )

            report.append(
                f"\nCandidate Answer:"
            )

            report.append(
                item["answer"]
            )

            report.append(
                f"\nScore: {item['score']}/10"
            )

            report.append(
                "\nFeedback:"
            )

            report.append(
                item["feedback"]
            )


        report.append(
            "\n\nTOPICS TO PRACTICE"
        )

        report.append(
            "-" * 30
        )

        for item in TOPIC_DATA[topic]["topics"]:

            report.append(
                f"- {item}"
            )


        final_report = "\n".join(
            report
        )


        st.download_button(
            "📥 Download Interview Report",
            final_report,
            file_name="AI_Interview_Report.txt",
            mime="text/plain"
        )


        # -----------------------------------------------
        # RESTART
        # -----------------------------------------------

        if st.button(
            "🔄 Start New Interview"
        ):

            st.session_state.questions = []
            st.session_state.current_question = 0
            st.session_state.evaluations = []
            st.session_state.answer_submitted = False
            st.session_state.interview_started = False

            st.rerun()


# ============================================================
# TAB 4 — AI ASSISTANT / RAG
# ============================================================

with tab4:

    st.header(
        "💬 AI Interview Assistant"
    )

    st.write(
        "Ask questions about the interview guide "
        "or your uploaded resume."
    )


    if (
        st.session_state.resume_vectorstore is None
        and
        st.session_state.guide_vectorstore is None
    ):

        st.info(
            "Upload a resume or ask questions about the interview guide."
        )


    user_query = st.text_input(
        "Ask your question:"
    )


    if st.button(
        "Ask AI"
    ) and user_query.strip():

        retrieved_context = []


        # -----------------------------------------------
        # GUIDE RETRIEVAL
        # -----------------------------------------------

        guide_docs = (
            st.session_state.guide_vectorstore
            .similarity_search(
                user_query,
                k=3
            )
        )

        for doc in guide_docs:

            retrieved_context.append(
                doc.page_content
            )


        # -----------------------------------------------
        # RESUME RETRIEVAL
        # -----------------------------------------------

        if (
            st.session_state.resume_vectorstore
            is not None
        ):

            resume_docs = (
                st.session_state.resume_vectorstore
                .similarity_search(
                    user_query,
                    k=2
                )
            )

            for doc in resume_docs:

                retrieved_context.append(
                    doc.page_content
                )


        context = "\n\n".join(
            retrieved_context
        )


        # -----------------------------------------------
        # RAG PROMPT
        # -----------------------------------------------

        rag_prompt = f"""
You are an AI Interview Preparation Assistant.

Answer the question using ONLY the provided context.

If the answer is not supported by the context,
say:

"I don't have enough information in the provided documents."

Context:
{context}

Question:
{user_query}

Give a short and clear answer.
"""


        try:

            response = (
                chat_model.invoke(
                    rag_prompt
                )
            )

            answer = response.content

            st.subheader(
                "🤖 Answer"
            )

            st.write(
                answer
            )


        except Exception as e:

            st.error(
                f"AI error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Interview Preparation Assistant | "
    "Python • LangChain • FAISS • Hugging Face • Qwen"
)