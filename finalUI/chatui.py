import streamlit as st
import random
import time
from util import get_parser
from LlamaChat import LlamaChat
from util import logger


def run_chat(llama:LlamaChat):


    # Accept user input
    if prompt := st.chat_input("Type your question here"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = get_output(llama=llama, qry=prompt)
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def main():
    global llama
    st.title("Legal Assistant")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    args = get_parser().parse_args()
    print(args.pretrained)
    llama = LlamaChat(args.pretrained)
    run_chat(llama=llama)

def get_output(llama:LlamaChat, qry:str)-> str:
        matching_docs = llama.chrm.similarity_search(qry,k=5)
        page_contents = [x.page_content for x in matching_docs]
        info = " ".join(page_contents)

        prompt = f"""<s>[INST]<<SYS>>
you are a respectful legal assistant who specializes in indian laws. Always answer the question based on information provided<</SYS>>

Information: 
{info}
Question:
{qry}
[/INST]
</s>
"""    
        output = llama.generate(prompt)
        #output = "fixed for now"
        return output

if __name__ == '__main__':
    main()