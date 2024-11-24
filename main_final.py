import streamlit as st
import re
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
import datetime
import os

load_dotenv()

def save_history(question, answer):
    with open('history.txt', 'a') as f:
        f.write(f'{datetime.datetime.now()}: {question} -> {answer}\n')


def load_history():
    if os.path.exists('history.txt'):
        with open('history.txt', 'r') as f:
            return f.readlines()
    return []


def main():
    st.set_page_config(page_title="Agente de Python Interactivo", page_icon='', layout="wide")

    st.title("Agente de Python Interactivo")
    st.markdown("""
        <style>
        .stApp{backgroud-color:black;}
        .title{color:#ff4b4b}
        .button{backgroud-color: #ff4b4b; color:white;border-radius: 5px;}
        .input{border: 1px solid #ff4b4b; border-radius: 5px;}
        </style>
        """, unsafe_allow_html=True)

    instrucciones = """
    - Siempre usa la herramienta, incluso si sabes la respuesta.
    - Puedes usar código de Python para responder.
    - Eres un agente que puede escribir código.
    - Si no sabes la respuesta, escribe "No sé la respuesta".
    """

    st.markdown(instrucciones)

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instrucciones=instrucciones)

    tools = [PythonREPLTool()]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agente = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agente_executor = AgentExecutor(
        agent=agente,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    st.markdown("### Python Agent:")
    ejemplos = [
        "Calcula la suma de 2 y 3.",
        "Genera una lista del 1 al 10.",
        "Crea una función que calcule el factorial de un número.",
    ]
    example = st.selectbox("Selecciona un ejemplo:", ejemplos)

    if st.button("Ejecutar"):
        user_input = example
        try:
            respuesta = agente_executor.invoke(input={"input": user_input, "instructions": instrucciones, "agent_scratchpad": ""})
            st.markdown("### Resultado")
            st.code(respuesta["output"], language="python")
            save_history(user_input, respuesta["output"])
        except ValueError as e:
            st.error(f"Error en el agente: {str(e)}")

    instructions = """ You are an agent designed to write and execute Python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer questions.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the questions, just return "I don't know" as the answer. 
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        path="username.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    st.subheader("Haz alguna pregunta")
    user_input = st.text_area("Escribe tu pregunta aquí:")

    def es_pregunta_csv(user_input):
        palabras_clave = ["cuántos", "promedio","username", "csv", "sumar", "contar", "máximo", "mínimo", "total", "mayor", "menor",
                          "ranking"]
        user_input = user_input.lower()

        for palabra in palabras_clave:
            if re.search(r'\b' + palabra + r'\b', user_input):
                return True
        return False

    if st.button("Ejecutar pregunta"):
        if es_pregunta_csv(user_input):
            try:
                respuesta = csv_agent.invoke(input={"input": user_input})
                st.markdown("### Resultado (CSV Agent)")
                st.write(respuesta["output"])
                save_history(user_input, respuesta["output"])
            except Exception as e:
                st.error(f"Error ejecutando el CSV Agent: {str(e)}")
        else:
            try:
                respuesta = agent_executor.invoke(input={"input": user_input, "instructions": instrucciones, "agent_scratchpad": ""})
                st.markdown("### Resultado (Python Agent)")
                st.code(respuesta["output"], language="python")
                save_history(user_input, respuesta["output"])
            except Exception as e:
                st.error(f"Error ejecutando el Python Agent: {str(e)}")


if __name__ == '__main__':
    main()
