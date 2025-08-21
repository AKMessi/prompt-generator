import streamlit as st
import os
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain import hub


# load api key
load_dotenv()

@st.cache_resource
def load_agent_components():
    """Loads all necessary components for the agent and caches them."""
    print("Loading agent components...")

    # load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    # load vectorstore
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    # load llm
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    # creating the retriever tool
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(
        retriever,
        "prompt_engineering_best_practices_retriever",
        "Searches and returns expert guidance on how to craft effective prompts."
    )
    tools = [retriever_tool]

    prompt_template = hub.pull("hwchase17/openai-tools-agent")
    prompt_template.messages[0].prompt.template = """
    You are an expert Prompt Architect. Your primary goal is to help users by creating new, high-quality prompts tailored to their specific needs.
    Your knowledge and methodology are based ENTIRELY on the principles within the 'Prompt Like A Pro' guide. You must apply the techniques from this guide to construct your final response.
    Follow this thought process meticulously:
    Step 1: The Interview
    First, engage the user in a short conversation to understand their goal. If their initial request is vague, you must ask clarifying questions about their target audience, desired output format, constraints, and any other relevant context. Do not proceed until you have a clear picture of their needs.
    Step 2: Consult the Guide
    Once you have enough context, use the `prompt_engineering_best_practices_retriever` tool to find the most relevant PRINCIPLES, INGREDIENTS, or TEMPLATES from the 'Prompt Like A Pro' guide. Your search query should be focused on the *methods* you need to use.
    Step 3: Architect the New Prompt
    This is the most critical step. You will synthesize a brand new prompt by APPLYING the techniques from the guide to the user's request.
    Your output must be the final, ready-to-use prompt inside a markdown code block.
    IMPORTANT: You are not searching the guide for a pre-written answer. You are searching it for the foundational rules and methods, which you will then use as a true expert to build a custom solution.
    """

    # create the agent
    agent = create_tool_calling_agent(llm, tools, prompt_template)

    return agent, tools

# streamlit app interface

st.set_page_config(page_title="Prompt Architect Agent", page_icon="🧑‍🎨")
st.title("🧑‍🎨 Prompt Architect Agent")
st.caption("Your AI-powered partner for crafting perfect prompts.")


agent, tools = load_agent_components()

if "agent_executor" not in st.session_state:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you craft a prompt today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What prompt can I help you architect?"):
    # Add the user's message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Consulting the guide and thinking..."):
        response = st.session_state.agent_executor.invoke({"input": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
    
    st.rerun()