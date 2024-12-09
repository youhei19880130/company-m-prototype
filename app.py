# 実行コマンドは、「streamlit run app.py」

import json
import boto3
from botocore.config import Config
import streamlit as st
import pandas as pd
from io import StringIO
import logging

# 再試行ポリシーを設定
config = Config(
    retries={
        'max_attempts': 4,  # 最大リトライ回数
        'mode': 'standard'   # リトライモード (standard または adaptive)
    }
)

class CFG:
    region="ap-northeast-1"
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    # system_prompt = "あなたは多くのデータにアクセス可能な経済学者です。"
    system_prompt = "あなたは資料のレビュー担当です。"
    temperature = 0.5
    top_k = 200
    knowledge_base_id = "WJQ1UWKXRG"

@st.cache_resource
def get_bedrock_client():
    return boto3.client(service_name="bedrock-runtime", region_name=CFG.region, config=config)

def generate_response(messages):
    print(messages)
    bedrock_client = get_bedrock_client()
    system_prompts = [{"text": CFG.system_prompt}]

    inference_config = {"temperature": CFG.temperature}
    additional_model_fields = {"top_k": CFG.top_k}

    response = bedrock_client.converse(
        modelId=CFG.model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields,
    )

    return response["output"]["message"]

def display_history(messages):
    for message in st.session_state.messages:
        display_msg_content(message)

def display_msg_content(message):
    print(message)
    with st.chat_message(message["role"]):
        st.write(message["content"])

def _invoke_model_with_response_stream_claude(input, message_placeholder, full_response):
    # Bedrockからのストリーミング応答を処理

    bedrock = boto3.client('iam', aws_access_key_id=st.secrets["AWS_ACCESS"], aws_secret_access_key=st.secrets["AWS_SECRET"], service_name="bedrock-runtime", region_name=CFG.region, config=config)
    messages = [m["role"] + ":" + m["content"] for m in st.session_state.messages]

    body = json.dumps(
        {
            "prompt": "\n\n" + "\n\n".join(messages) + "\n\nAssistant:",
            "max_tokens_to_sample": st.session_state["max_tokens_to_sample"],
            "temperature": st.session_state["temperature"],
            "top_p": st.session_state["top_p"],
        }
    )

    response = bedrock.invoke_model_with_response_stream(modelId=st.session_state["bedrock_model"], body=body)
    stream = response.get("body")
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                full_response += json.loads(chunk.get("bytes").decode())["completion"]
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    return full_response

def _retrieve_and_generate(input, message_placeholder):
    # BedrockからのRAG応答を処理

    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", aws_access_key_id=st.secrets["AWS_ACCESS"], aws_secret_access_key=st.secrets["AWS_SECRET"], region_name=CFG.region, config=config)
    full_response = bedrock_agent_runtime.retrieve_and_generate(
        input={"text": input},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": st.session_state["knowledge_base_id"],
                "modelArn": f'arn:aws:bedrock:{CFG.region}::foundation-model/{st.session_state["bedrock_model"]}',
            },
        },
    )["output"]["text"]

    message_placeholder.markdown(full_response)
    return full_response

def main():
    st.set_page_config(page_title="ChatBot", page_icon="🤗")
    # サイドバーを表示
    st.sidebar.title("基盤モデル設定")
    st.session_state["bedrock_model"] = CFG.model_id
    st.session_state["knowledge_base_id"] = CFG.knowledge_base_id
    st.session_state["rag_on"] = True
    
    # 左サイドバー
    # with st.sidebar:
    #     # bedrock_model = st.selectbox("Bedrockのモデルを選択してください", (BEDROCK_MODEL_LIST))
    #     st.session_state["bedrock_model"] = CFG.model_id
    
    #     with st.expander(label="RAG", expanded=False):
    #         rag_on = st.toggle("Knowledge base")
    #         if rag_on:
    #             # st.session_state["knowledge_base_id"] = st.text_input(label="Knowledge base ID", type="default")
    #             st.session_state["knowledge_base_id"] = CFG.knowledge_base_id
    #         st.session_state["rag_on"] = rag_on
    
    #     with st.expander(label="Configurations", expanded=True):
    #         max_tokens_to_sample = st.slider(label="Maximum length", min_value=0, max_value=2048, value=300)
    #         st.session_state["max_tokens_to_sample"] = max_tokens_to_sample
    
    #         temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    #         st.session_state["temperature"] = temperature
    
    #         top_p = st.slider(label="top_p", min_value=0.00, max_value=1.00, value=0.90, step=0.01)
    #         st.session_state["top_p"] = top_p

    st.title("Bedrock Converse API Chatbot")
    
    uploaded_file = st.file_uploader("Choose a pdf file", type='pdf')
    if uploaded_file is not None:
        document = {"document": {"format": "pdf", "name": "string", "source": {"bytes": uploaded_file.read()}}}

        # To read file as bytes:
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        # To convert to a string based IO:
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    display_history(st.session_state.messages)

    if prompt := st.chat_input("What's up?"):
        # input_msg = {"role": "user", "content": [{"text": prompt}, document]}
        # display_msg_content(input_msg)
        # st.session_state.messages.append(input_msg)

        # response_msg = generate_response(st.session_state.messages)
        # display_msg_content(response_msg)
        # st.session_state.messages.append(response_msg)
        st.session_state.messages.append({"role": "Human", "content": prompt})
        with st.chat_message("Human"):
            st.markdown(prompt)
    
        with st.chat_message("Assistant"):
            with st.spinner("回答を生成中..."):
                message_placeholder = st.empty()
                full_response = ""
    
                if st.session_state["rag_on"]:
                    full_response = _retrieve_and_generate(prompt, message_placeholder)
                else:
                    # full_response = generate_response(prompt)
                    full_response = _invoke_model_with_response_stream_claude(prompt, message_placeholder, full_response)
                
        st.session_state.messages.append({"role": "Assistant", "content": full_response})
        st.session_state.Clear = True  # チャット履歴のクリアボタンを有効にする

if __name__ == "__main__":
    main()

