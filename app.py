import streamlit as st
import pandas as pd
import re
import tempfile
import os

# ---- LangChain & LLM imports ----
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chat_models import ChatOpenAI,  ChatAnthropic
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -------------------------------------------------------------------
# 1. Utility functions for building a hierarchical tree from ref_id
# -------------------------------------------------------------------
def natural_sort_key(ref_id: str):
    """
    A 'natural' sort key that splits the string into digit and non-digit chunks.
    Each chunk is turned into a 2-element tuple:
      - (0, 'lowercased_text') for text chunks
      - (1, integer) for numeric chunks
    Example: "Article 10" -> [ (0, 'article '), (1, 10) ]
    """
    parts = re.split(r'(\d+)', ref_id)
    sort_key = []
    for part in parts:
        if not part:  # skip empty
            continue
        if part.isdigit():
            sort_key.append((1, int(part)))
        else:
            sort_key.append((0, part.lower()))
    return sort_key

def get_parent_ref_id(ref_id: str) -> str:
    parts = ref_id.split(".")
    if len(parts) > 1:
        return ".".join(parts[:-1])
    else:
        return ""

def build_tree(df: pd.DataFrame) -> dict:
    """
    Build a tree (dict) from DataFrame rows with 'ref_id', 'name', 'description'.
    """
    tree_nodes = {}
    for _, row in df.iterrows():
        ref_id = str(row["ref_id"])
        tree_nodes[ref_id] = {
            "name": row.get("name", "Unnamed"),
            "description": row.get("description", "No description available."),
            "children": {}
        }
    # Link child nodes to parents
    roots = {}
    for ref_id, node_data in tree_nodes.items():
        parent_id = get_parent_ref_id(ref_id)
        if parent_id == "":
            roots[ref_id] = node_data
        else:
            parent_node = tree_nodes.get(parent_id)
            if parent_node:
                parent_node["children"][ref_id] = node_data
            else:
                # Parent not found => treat as root
                roots[ref_id] = node_data
    return roots

def render_tree(tree_dict: dict, indent: int = 0) -> str:
    """
    Recursively render the tree as an indented bullet list in Markdown,
    using the natural_sort_key to compare ref_ids.
    """
    lines = []
    indent_spaces = " " * indent

    def sort_key(item):
        return natural_sort_key(item[0])

    for ref_id, node_data in sorted(tree_dict.items(), key=sort_key):
        name = node_data["name"]
        description = node_data["description"]

        lines.append(f"{indent_spaces}- **{ref_id}**: {name}")
        desc_indent_spaces = " " * (indent + 4)
        lines.append(f"{desc_indent_spaces}{description}")

        if node_data["children"]:
            child_output = render_tree(node_data["children"], indent + 4)
            lines.append(child_output)

    return "\n".join(lines)

def show_ai_act_hierarchy(df: pd.DataFrame):
    """
    Show the AI Act data in a hierarchical view, using *natural sort*.
    """
    if df.empty:
        st.warning("No AI Act data available to display.")
        return

    st.subheader("AI Act Hierarchical View (Natural Ordering)")

    tree = build_tree(df)

    def sort_key(item):
        return natural_sort_key(item[0])

    for root_id, root_data in sorted(tree.items(), key=sort_key):
        with st.expander(f"{root_id}: {root_data['name']}", expanded=False):
            st.write(root_data["description"])
            if root_data["children"]:
                subtree_markdown = render_tree(root_data["children"], indent=0)
                st.markdown(subtree_markdown, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. Data Loading and Preprocessing
# -------------------------------------------------------------------
def load_data(filepath: str) -> dict:
    excel_data = pd.ExcelFile(filepath)
    return {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}

def preprocess_cra_data(dataframes: dict) -> pd.DataFrame:
    cra_df = dataframes.get("CRA", pd.DataFrame())
    if not cra_df.empty:
        cra_df = cra_df.rename(columns=str.lower)
        required_cols = {"ref_id", "depth", "description", "name"}
        missing_cols = required_cols - set(cra_df.columns)
        for col in missing_cols:
            cra_df[col] = None
        cra_df["description"].fillna("No description available.", inplace=True)
        cra_df["name"].fillna("Unnamed Category", inplace=True)
        cra_df["ref_id"] = cra_df["ref_id"].astype(str)
    return cra_df

def preprocess_masvs_data(dataframes: dict) -> pd.DataFrame:
    masvs_df = dataframes.get("masvs", pd.DataFrame())
    if not masvs_df.empty:
        if "description" not in masvs_df.columns:
            masvs_df["description"] = "No description available."
        masvs_df["description"].fillna("No description available.", inplace=True)
    return masvs_df

def preprocess_ai_act_data(dataframes: dict) -> pd.DataFrame:
    ai_act_df = dataframes.get("AI ACT", pd.DataFrame())
    if not ai_act_df.empty:
        ai_act_df = ai_act_df.rename(columns=str.lower)
        required_cols = {"depth", "ref_id", "name", "description"}
        missing_cols = required_cols - set(ai_act_df.columns)
        for col in missing_cols:
            ai_act_df[col] = None
        ai_act_df["description"].fillna("No description available.", inplace=True)
        ai_act_df["name"].fillna("Unnamed Section", inplace=True)
        ai_act_df["ref_id"] = ai_act_df["ref_id"].astype(str)
    return ai_act_df

# -------------------------------------------------------------------
# 3. Fetching Hierarchies for CRA
# -------------------------------------------------------------------
def fetch_hierarchy(df: pd.DataFrame, main_ref: str) -> pd.DataFrame:
    if "ref_id" not in df.columns or "depth" not in df.columns:
        raise ValueError("The DataFrame is missing 'ref_id' or 'depth'.")
    df["ref_id"] = df["ref_id"].astype(str)
    subset = df[df["ref_id"].str.startswith(main_ref, na=False)]
    subset = subset.sort_values(by="ref_id")
    return subset

# -------------------------------------------------------------------
# 4. Multi-LLM Selection Helper
# -------------------------------------------------------------------
def get_llm(model_name: str, openai_api_key: str = None, anthropic_api_key: str = None, huggingfacehub_api_key: str = None):
    if model_name == "GPT-3.5":
        return ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
    elif model_name == "GPT-4":
        return ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
    elif model_name == "Claude 2":
        return ChatAnthropic(
            temperature=0,
            anthropic_api_key=anthropic_api_key,
            model_name="claude-v2"
        )
    elif model_name == "Mistral Large":
        return HuggingFaceHub(
            repo_id="mistralai/mistral-large",
            huggingfacehub_api_token=huggingfacehub_api_key,
            model_kwargs={"temperature": 0}
        )
    else:
        # Fallback to GPT-3.5 if an unknown option is selected.
        return ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
    """
    Returns a LangChain LLM or Chat model instance 
    depending on the user's choice in the sidebar.
    """
    if model_name == "GPT-3.5":
        return ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
    elif model_name == "GPT-4":
        return ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
    elif model_name == "Claude 2":
        # Placeholder for Anthropic's Claude model
        return OpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo"  # fallback
        )
    elif model_name == "Mistral Large":
        # Placeholder for a HF or local model
        return OpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo"  # fallback
        )
    else:
        return ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )

# -------------------------------------------------------------------
# 5. Create a Pandas DF Agent for Excel Data
# -------------------------------------------------------------------
def create_agent(df: pd.DataFrame, llm, verbose=False):
    """
    Modified to accept an LLM object directly.
    We'll use that LLM in the create_pandas_dataframe_agent.
    """
    return create_pandas_dataframe_agent(llm, df, verbose=verbose, allow_dangerous_code=True)

# -------------------------------------------------------------------
# 6. PDF/Text File Q&A Helpers
# -------------------------------------------------------------------
def build_vectorstore_from_docs(docs, openai_api_key: str):
    """
    Takes a list of Document objects, splits them into chunks,
    and builds a FAISS vector store using OpenAI embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splitted_docs = []
    for doc in docs:
        for chunk in text_splitter.split_text(doc.page_content):
            splitted_docs.append(
                Document(page_content=chunk, metadata=doc.metadata)
            )

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(splitted_docs, embeddings)
    return vectorstore

def handle_pdf_text_uploads(openai_api_key: str, selected_llm_name: str):
    """
    Let user upload PDFs or TXT files, build a vectorstore, and
    allow conversational Q&A with the chosen LLM.
    """
    st.subheader("Upload PDF or Text files for Q&A")
    # Added a unique key to avoid duplicate element IDs
    uploaded_files = st.file_uploader(
        "Upload one or more PDF or Text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="pdf_text_upload_key"
    )

    if "file_qa_chain" not in st.session_state:
        st.session_state["file_qa_chain"] = None

    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            # Save file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            extension = os.path.splitext(uploaded_file.name)[1].lower()
            if extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                file_docs = loader.load()  # List of Document objects
            else:
                # For .txt files
                loader = TextLoader(temp_file_path, encoding="utf-8")
                file_docs = loader.load()

            # Optionally remove the temporary file after processing
            os.remove(temp_file_path)

            all_docs.extend(file_docs)

        # Build a vector store from the uploaded documents
        vectorstore = build_vectorstore_from_docs(all_docs, openai_api_key)
        # Get the selected LLM
        llm = get_llm(selected_llm_name, openai_api_key)
        # Create a retriever from the vectorstore (retrieves top 3 matching chunks)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Set up conversation memory for multi-turn interactions
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Create a ConversationalRetrievalChain for interactive Q&A
        file_qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

        st.session_state["file_qa_chain"] = file_qa_chain
        st.success("Vector store created for uploaded files. You can now ask questions below.")

    if st.session_state["file_qa_chain"]:
        question = st.text_input("Ask a question about your uploaded files:")
        if st.button("Ask (Files)"):
            if question.strip():
                with st.spinner("Generating answer..."):
                    result = st.session_state["file_qa_chain"]({"question": question})
                    answer = result.get("answer", "No answer generated.")
                    st.markdown(f"**Answer:** {answer}")
                    
                    # Optionally display the conversation history
                    chat_history = result.get("chat_history", [])
                    with st.expander("Chat History"):
                        for msg in chat_history:
                            # Try to access role and content; if not present, fall back.
                            try:
                                role = msg.role
                            except AttributeError:
                                # Fallback: infer role from class name
                                if msg.__class__.__name__ == "HumanMessage":
                                    role = "human"
                                elif msg.__class__.__name__ == "AIMessage":
                                    role = "assistant"
                                else:
                                    role = "unknown"
                            try:
                                content = msg.content
                            except AttributeError:
                                content = str(msg)
                            st.markdown(f"**{role.title()}:** {content}")
            else:
                st.warning("Please enter a question.")

# -------------------------------------------------------------------
# 7. Main Streamlit App (Combined)
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Scheme AI App", layout="wide")
    st.title("Scheme AI: Excel Datasets + PDF/Text Q&A")
    st.write("Choose a data source in the sidebar. Also pick which LLM you want to use.")

    # ---- Sidebar Configuration ----
    st.sidebar.title("Configuration")
    model_choice = st.sidebar.selectbox(
        "LLM Model",
        ["GPT-3.5", "GPT-4", "Claude 2", "Mistral Large"],
        index=0
    )
    
    # Show the appropriate API key input field based on the selected model.
    if model_choice in ["GPT-3.5", "GPT-4"]:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        anthropic_api_key = None
        huggingfacehub_api_key = None
    elif model_choice == "Claude 2":
        anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
        openai_api_key = None
        huggingfacehub_api_key = None
    elif model_choice == "Mistral Large":
        huggingfacehub_api_key = st.sidebar.text_input("HuggingFace Hub API Key", type="password")
        openai_api_key = None
        anthropic_api_key = None

    data_source = st.sidebar.selectbox("Data Source", ["Excel Datasets", "PDF/Text Files"])

    # Check if the required API key has been provided.
    if (model_choice in ["GPT-3.5", "GPT-4"] and not openai_api_key) or \
       (model_choice == "Claude 2" and not anthropic_api_key) or \
       (model_choice == "Mistral Large" and not huggingfacehub_api_key):
        st.warning("Please provide the appropriate API key for the selected model in the sidebar.")
        return

    # ================= Excel Datasets Flow =================
    if data_source == "Excel Datasets":
        # Let the user pick from the available Excel datasets
        dataset_options = {
            "OWASP MASVS v2.1.0": "owasp-masvs-v2.1.0.xlsx",
            "CRA Regulation Annexes": "cra-regulation-annexes.xlsx",
            "AI Act Directive": "AI-act.xlsx"
        }
        dataset_choice = st.selectbox("Select a Dataset", list(dataset_options.keys()))
        data_file = dataset_options.get(dataset_choice)

        # Load the data from the selected file
        try:
            dataframes = load_data(data_file)
            if dataset_choice == "OWASP MASVS v2.1.0":
                df = preprocess_masvs_data(dataframes)
            elif dataset_choice == "CRA Regulation Annexes":
                df = preprocess_cra_data(dataframes)
            elif dataset_choice == "AI Act Directive":
                df = preprocess_ai_act_data(dataframes)
            else:
                df = pd.DataFrame()
            st.write(f"Loaded dataset: **{dataset_choice}** with sheets: {list(dataframes.keys())}")
        except Exception as e:
            st.error(f"Failed to load dataset. Error: {e}")
            return

        if df.empty:
            st.error("The chosen dataset is empty or could not be loaded.")
            return

        # --- Dataset-specific UI ---
        if dataset_choice == "CRA Regulation Annexes":
            main_ref = st.text_input("Enter a main requirement ID (e.g. '1', '1.1')").strip()
            if main_ref:
                try:
                    filtered_hierarchy = fetch_hierarchy(df, main_ref)
                    if not filtered_hierarchy.empty:
                        st.write(f"Requirements under '{main_ref}':")
                        for _, row in filtered_hierarchy.iterrows():
                            st.markdown(f"**{row['ref_id']}**: {row['description']}")
                        st.table(filtered_hierarchy)
                    else:
                        st.warning(f"No requirements found for '{main_ref}'.")
                except Exception as e:
                    st.error(f"Error fetching hierarchy: {e}")

        if dataset_choice == "AI Act Directive":
            depth = st.number_input("Enter depth level (e.g., 1, 2):", min_value=1, step=1)
            if depth:
                try:
                    filtered = df[df["depth"] == depth].sort_values(by="ref_id")
                    if not filtered.empty:
                        st.write(f"Sections at depth **{depth}**:")
                        st.table(filtered[["ref_id", "name", "description"]])
                    else:
                        st.warning(f"No sections found for depth '{depth}'.")
                except Exception as e:
                    st.error(f"Error fetching AI Act by depth: {e}")
            # Also display the hierarchical view using natural ordering
            show_ai_act_hierarchy(df)

        # Create the LLM instance using all relevant API keys.
        llm = get_llm(model_choice, openai_api_key, anthropic_api_key, huggingfacehub_api_key)
        agent = create_agent(df, llm=llm, verbose=False)

        user_question = st.text_input("Ask a question about this Excel data:")
        if st.button("Ask (Excel)"):
            if user_question.strip():
                with st.spinner("Generating answer..."):
                    try:
                        answer = agent.invoke(user_question)
                        # Answer may be a DataFrame, dict, or string.
                        if isinstance(answer, pd.DataFrame):
                            st.dataframe(answer)
                        elif isinstance(answer, dict) and "output" in answer:
                            st.markdown(answer["output"])
                        else:
                            st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a valid question.")

    # ================= PDF/Text Files Flow =================
    else:
        # In the PDF/Text flow, we'll reuse the handle_pdf_text_uploads helper.
        # Note: You may wish to update that helper to pass the additional API keys.
        # For simplicity, here we pass the OpenAI key if available (for GPT models)
        # or rely on the model_choice within the helper.
        handle_pdf_text_uploads(openai_api_key, model_choice)

if __name__ == "__main__":
    main()