import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader

# Ollama 언어 모델 서버의 기본 URL
CUSTOM_URL = "http://localhost:11434"

# Streamlit 앱의 레이아웃과 제목 구성
# st.set_page_config(layout="wide", page_title="🦜🔗 Summarization App")
st.title(" 🦜 PDF을 요약해드려요")

# 영어 요약을 위한 Ollama 언어 모델 초기화
llm = Ollama(
    model="bnksys/yanolja-eeve-korean-instruct-10.8b:latest", 
    base_url=CUSTOM_URL, 
    temperature=0,    
    num_predict=200
)

# PDF 파일을 읽고 처리하기 위한 리소스 캐시
@st.cache_resource
def read_file(file_name):
    """
    PDF 파일을 읽고 관리 가능한 청크로 나눕니다.

    Args:
        file_name (UploadedFile): 업로드된 PDF 파일.

    Returns:
        list: 문서 청크 리스트.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file_name.getbuffer())
        file_path = tf.name
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def summarize_documents(txt_input):
    """
    업로드된 문서를 요약하고 요약을 표시합니다.

    Args:
        txt_input (list): 요약할 문서 청크 리스트.
    """
    map_prompt_template = """
    - 당신은 전문 요약가입니다.
    - 제공된 텍스트의 간결한 요약을 만들어 주세요.
    - 요약만 응답해 주세요.
    {text}
    """
    summary_result = ""
    message_placeholder = st.empty()
    
    for doc in txt_input:
        prompt_text = map_prompt_template.format(text=doc)
        stream_generator = llm.stream(prompt_text)
        
        for chunk in stream_generator:
            summary_result += chunk
            message_placeholder.markdown(summary_result)

def main():
    """
    Streamlit 앱을 실행하는 메인 함수.
    """
    if 'summary_result' not in st.session_state:
        st.session_state.summary_result = ""

    st.markdown("#### PDF 업로드 ▼ ")
    uploaded_file = st.file_uploader('pdfuploader', label_visibility="hidden", accept_multiple_files=False, type="pdf")
    
    if uploaded_file is not None:
        txt_input = read_file(uploaded_file)
        with st.spinner("문서를 요약하는 중..."):
            summarize_documents(txt_input)

if __name__ == "__main__":
    main()
