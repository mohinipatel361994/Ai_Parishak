import json
from langchain.llms import OpenAI
import chat
import openai
from openai import OpenAI
import streamlit as st
import os

import pandas as pd
#from rag import *
from PIL import Image
from chat import load_chain
import numpy as np
from prompts import initialise_prompt
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import docx
from prompts import ai_prompt,ai_topic_prompt,latex_prompt
from functions import read_word_table,create_word_doc,get_text
import subprocess
import re
from dotenv import load_dotenv
from docxlatex import Document
import json
import base64
import fixed_function
from io import BytesIO
#from docxlatex import Document
# from PIL import Image
#client = OpenAI()

st.set_page_config(layout='wide')

hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            h1 {
               color: green;
               font-size: 35px;    
               }
            
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
add_selectbox = st.sidebar.selectbox(
    "Store the chat hsitory",
    ("Teachers", "Students", "Administration")
)

load_dotenv()

images = ['6MarkQ']

#os.environ["OPENAI_API_TYPE"] = ""
#os.environ["OPENAI_API_VERSION"] = ""
#os.environ["OPENAI_API_BASE"] = ""
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if 'OPENAI_API_KEY' in st.secrets:
    st.success('API key already provided!', icon='✅')
    replicate_api = st.secrets['OPENAI_API_KEY']
else:
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
       if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
          st.warning('Please enter your credentials!', icon='⚠️')
       else:
          st.success('Proceed to entering your prompt message!', icon='👉')

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return  base64.b64encode(data).decode()



def download_doc(doc):
    doc_buffer = BytesIO()
    doc.save(doc_buffer)
    doc_buffer.seek(0)
    return doc_buffer


with open('style.css') as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 


def mcq_to_markdown(mcq_dict):
    markdown = ""
    for key, value in mcq_dict.items():
        if key.isdigit():
            markdown += f"Question {key}: {value}\n"
        elif key.startswith('image'):
            markdown += f"![Image {key[-1]}]({value})\n\n"
    return markdown


def display_mcq(mcq_dict):
    for key, value in mcq_dict.items():
        st.markdown(f"""**Question {key}:** {value['question']}""")
        c1,c2 = st.columns([3,1])
        with c1:
            st.write("Options:")
            for option_key, option_value in value['options'].items():
                st.write(f"{option_key}. {option_value}")
        with c2:
            st.image(f"{value['image']}", use_column_width=True)

def is_word_in_text(word, text):
    """
    Check if a word is within a given text.

    Args:
    - word (str): The word to check for.
    - text (str): The text to search within.

    Returns:
    - bool: True if the word is found in the text, False otherwise.
    """
    # Split the text into individual words
    words_in_text = text.split()

    # Check if the word is in the list of words from the text
    if word in words_in_text:
        return True
    else:
        return False
    
def text_to_latex(question):
    # Convert question to LaTeX format
    latex_question = r"\text{" + question + r"}"
    return f"{latex_question}"
    
def add_dollar_signs(text):
    # Regular expression pattern to find equations
    equation_pattern = r"([+-]?\s*\d*\s*[xX]\^\d+)"

    # Find equations in the text using regular expression
    equations = re.findall(equation_pattern, text)

    # Replace each equation with the same equation surrounded by dollar signs
    for equation in equations:
        text = text.replace(equation, f"${equation}$")

    return text
    
def generate_quiz(text):

    test_source_text = text
    quiz_rag = RAGTask(task_prompt_builder=revision_quiz_json_builder)
#     summary_rag = RAGTask(task_prompt_builder=plaintext_summary_builder)
#     glossary_rag = RAGTask(task_prompt_builder=get_glossary_builder)

    outputs = []
    # for rag_task in [quiz_rag, summary_rag, glossary_rag]:
    #     output = rag_task.get_output(source_text=test_source_text)
    #     outputs.append(output)
    for rag_task in [quiz_rag]:
        output = rag_task.get_output(source_text=test_source_text)
        outputs.append(output)
    return outputs

def decrement_question_num():
    if st.session_state['curr_question'] > 0:
        st.session_state['curr_question'] -= 1
        #ClearAll()

def increment_question_num():
    print('Incrementing question', st.session_state['curr_question'])
    if st.session_state['curr_question'] < st.session_state['quiz_length'] - 1:
        st.session_state['curr_question'] += 1
        #ClearAll()
    


def display_q(df):
    for index,row in df.iterrows():
        st.markdown("##### **Question "+str(index+1)+"**")
        if row['Image Path']=='no_image':
            # st.session_state.format_chain = ConversationChain( llm=AzureChatOpenAI(
            # deployment_name="gpt-4turbo",
            # temperature=0
            # ))
            # formatted_question = chat.format_response(row['Question'])
            # st.latex(formatted_question)
            st.info(row['Question'])
        if row['Image Path'] !='no_image':
            # st.session_state.format_chain = ConversationChain( llm=AzureChatOpenAI(
            # deployment_name="gpt-4turbo",
            # temperature=0
            # ))
            # formatted_question = chat.format_response(row['Question'])
            # st.latex(formatted_question)
            st.info(row['Question'])
            st.image(row['Image Path'],width=180)
        st.divider()

pd.set_option('display.max_colwidth', None)
st.session_state.ques = pd.read_csv('update_question.csv')



with st.container():
    col1, col2, col3 = st.columns([0.2, 0.6, 0.2], gap="small")
    with col1:
        logo_image = Image.open('assests/madhya-pradesh-logo.png')
        resized_logo = logo_image.resize((150, 150))
        st.image(resized_logo)
    with col2:
        st.markdown("# AI Parikshak")
        st.markdown("###### AI Based Question Generation Assistance")
    with col3:
        l = Image.open('assests/28072020125926mpsedclogo.png')
        re = l.resize((165, 155))  # Corrected the resize method call
        st.image(re)
         
        #st.markdown("# Teacher Assistance")
#Creating the chatbot interface
#st.markdown("###### :disguised_face: :robot_face:  in•quis•i•tive | inˈkwizədiv, iNGˈkwizədiv  :robot_face: :disguised_face:")


def on_text_area_change():
    st.session_state.page_text = st.session_state.my_text_area

def read_pdf_page(file, page_number):
    pdfReader = PdfReader(file)
    page = pdfReader.pages[page_number]
    return page.extract_text()




def markdown_to_pdf(markdown: str, output_file: str):
    """
    Convert Markdown to PDF
    :param markdown: Markdown string
    :param output_file: Output file
    """
    with open(TEMP_MD_FILE, "w",encoding='utf-8') as f:
        f.write(markdown)


#english_to_punjabi_reference = {'natural numbers': 'ਪ੍ਰਾਕ੍ਰਿਤਿਕ ਸੰਖਿਆਵਾਂ', 
#                                 'whole numbers': 'ਪੂਰਨ ਸੰਖਿਆਵਾਂ', 
#                                 'integers': 'ਸੰਪੂਰਨ ਸੰਖਿਆਵਾਂ',
#                                 'number line': 'ਸੰਖਿਆ ਰੇਖਾ', 
#                                 'even numbers': 'ਜਿਸਤ ਸੰਖਿਆਵਾਂ', 
#                                 'odd numbers ': 'ਟਾਂਕ ਸੰਖਿਆਵਾਂ', 
#                                 'prime numbers': 'ਅਭਾਜ ਸੰਖਿਆਵਾਂ', 
#                                 'composite numbers': 'ਭਾਜ ਸੰਖਿਆਵਾਂ', 
#                                 'addition': 'ਜੋੜ', 
#                                 'subtraction': 'ਘਟਾਉ', 
#                                 'multiplication': 'ਗੁਣਾ', 
#                                 'division': 'ਭਾਗ', 
#                                 'fraction': 'ਭਿੰਨ', 
#                                 'proper fractions': 'ਉਚਿਤ ਭਿੰਨਾਂ', 
#                                 'improper fractions': 'ਅਣਉਚਿਤ ਭਿੰਨਾਂ', 
#                                 'mixed fractions': 'ਮਿਸ਼ਰਿਤ ਭਿੰਨਾਂ', 
#                                 'comparison': 'ਤੁਲਨਾ', 
#                                 'equivalent fractions': 'ਤੁੱਲ ਭਿੰਨਾਂ', 
#                                 'like fractions': 'ਸਮਾਨ ਭਿੰਨਾਂ', 
#                                 'unlike fractions': 'ਅਸਮਾਨ ਭਿੰਨਾਂ', 
#                                 'rational numbers': 'ਪਰਿਮੇਯ ਸੰਖਿਆਵਾਂ', 
#                                 'reciprocal': 'ਗੁਣਾਤਮਕ ਉਲਟ', 
#                                 'factor': 'ਗੁਨਣਖੰਡ', 
#                                 'multiple': 'ਗੁਣਜ', 
#                                 'place value': 'ਸਥਾਨਕ ਮੁੱਲ', 
#                                 'estimation': 'ਅੰਦਾਜ਼ਾ', 
#                                 'successor': 'ਅਗੇਤਰ', 
#                                 'predecessor': 'ਪਿਛੇਤਰ', 
#                                 'percentage': 'ਪ੍ਰਤੀਸ਼ਤ', 
#                                 'decimal': 'ਦਸ਼ਮਲਵ', 
#                                 'units': 'ਇਕਾਈਆਂ', 
#                                 'plane': 'ਤਲ', 
#                                 'point': 'ਬਿੰਦੂ', 
#                                 'ray': ' ਕਿਰਨ', 
#                                 'line': 'ਰੇਖਾ', 
#                                 'line segment': 'ਰੇਖਾਖੰਡ', 
#                                 'intersecting lines': 'ਕਾਟਵੀਆਂ ਰੇਖਾਵਾਂ', 
#                                 'concurrent lines': 'ਸੰਗਾਮੀ ਰੇਖਾਵਾਂ', 
#                                 'parallel lines': 'ਸਮਾਂਤਰ ਰੇਖਾਵਾਂ', 
#                                 'transversal': 'ਕਾਟਵੀ ਰੇਖਾ', 
#                                 'angle': 'ਕੋਣ', 
#                                 'acute angle': 'ਨਿਊਨ ਕੋਣ', 
#                                 'right angle': 'ਸਮਕੋਣ', 
#                                 'obtuse angle': 'ਅਧਿਕ ਕੋਣ', 
#                                 'straight angle': 'ਸਰਲ ਕੋਣ', 
#                                 'reflex angle': 'ਰਿਫਲੈਕਸ ਕੋਣ', 
#                                 'complementary angles': 'ਪੂਰਕ ਕੋਣ', 
#                                 'supplementary angles': 'ਸੰਪੂਰਕ ਕੋਣ', 
#                                 'linear pair': 'ਰੇਖੀ ਜੋੜਾ', 
#                                 'corresponding angles': 'ਸੰਗਤ ਕੋਣ', 
#                                 'vertically opposite angles': 'ਸਿਖਰ ਸਨਮੁੱਖ ਕੋਣ', 
#                                 'alternate angles': 'ਇਕਾਂਤਰ ਕੋਣ', 
#                                 'polygons': 'ਬਹੁਭੁਜ', 
#                                 'concave polygons': 'ਉੱਤਲ ਬਹੁਭੁਜ', 
#                                 'convex polygons': 'ਅਵਤਲ ਬਹੁਭੁਜ', 
#                                 'curve': 'ਵਕਰ', 
#                                 'diagonals': 'ਵਿਕਰਣ', 
#                                 'triangle': 'ਤਿਕੋਣ', 
#                                 'scalene triangle': 'ਬਿਖਮਭੁਜੀ ਤਿਕੋਣ',
#                                 'triangle':'ਤਿਕੋਣ', 
#                                 'isosceles triangle': 'ਸਮਦੋਭੁਜੀ ਤਿਕੋਣ', 
#                                 'equilateral triangle': 'ਸਮਭੁਜੀ ਤਿਕੋਣ', 
#                                 'acute angle triangle': 'ਨਿਊਨਕੋਣੀ ਤਿਕੋਣ', 
#                                 'obtuse angle triangle': 'ਅਧਿਕਕੋਣੀ ਤਿਕੋਣ', 
#                                 'right angle triangle': 'ਸਮਕੋਣੀ ਤਿਕੋਣ', 
#                                 'congruency': 'ਸਰਬੰਗਸਮਤਾ', 
#                                 'quadrilaterals': 'ਚਤੁਰਭੁਜ', 
#                                 'rectangle': 'ਆਇਤ', 
#                                 'square': 'ਵਰਗ', 
#                                 'parallelograms': 'ਸਮਾਂਤਰ ਚਤੁਰਭੁਜ', 
#                                 'trapezium': 'ਸਮਲੰਬ ਚਤੁਰਭੁਜ', 
#                                 'rhombus': 'ਸਮਚਤੁਰਭੁਜ', 
#                                 'kite': 'ਪਤੰਗ', 
#                                 'circle': 'ਚੱਕਰ', 
#                                 'centre': 'ਕੇਂਦਰ', 
#                                 'radius': 'ਅਰਧ ਵਿਆਸ', 
#                                 'diameter': 'ਵਿਆਸ', 
#                                 'circumference': 'ਘੇਰਾ', 
#                                 'perimeter': 'ਪਰਿਮਾਪ', 
#                                 '2-dimensional figures': '2-ਪਸਾਰੀ ਆਕ੍ਰਿਤੀਆਂ', 
#                                 '3-dimensional solids': '3-ਪਸਾਰੀ ਠੋਸ', 
#                                 'cube': 'ਘਣ', 
#                                 'cuboid': 'ਘਣਾਵ', 
#                                 'cylinder': 'ਵੇਲਣ', 
#                                 'cone': 'ਸ਼ੰਕੂ', 
#                                 'sphere': 'ਗੋਲਾ', 
#                                 'hemisphere': 'ਅਰਧ ਗੋਲਾ', 
#                                 'faces': 'ਫਲਕ', 
#                                 'vertex': 'ਸ਼ਿਖਰ', 
#                                 'edges': 'ਕਿਨਾਰੇ', ',
#                                 'square root': 'ਵਰਗਮੂਲ', 
#                                 'ratio': 'ਅਨੁਪਾਤ', 
#                                 'proportion': 'ਸਮਾਨ ਅਨੁਪਾਤ', 
#                                 'pi  (π)': 'ਪਾਈ (π)', 
#                                 'symmetry': 'ਸਮਮਿਤੀ', 
#                                 'variable': 'ਚਲ', 
#                                 'constant': 'ਸਥਿਰ ਅੰਕ', ,
#                                 'algebraic expressions': 'ਬੀਜਗਣਿਤਕ ਵਿਅੰਜਕ', '',
#                                 'algebraic identities': 'ਬੀਜਗਣਿਤਕ ਤਤਸਮਕ', ,
#                                 'pythagorean triplet': 'ਪਾਈਥਾਗੋਰੀਅਨ ਤ੍ਰਿਗੁਟ', '
#                                 'like terms': 'ਸਮਾਨ ਪਦ', 
#                                 'unlike terms': 'ਅਸਮਾਨ ਪਦ', 
#                                 'LCM': 'ਲ.ਸ.ਵ.', 
#                                 'HCF': 'ਮ.ਸ.ਵ',
#                                 'polynomial':'ਬਹੁਪਦ',
#                                 "whole numbers": "ਪੂਰਨ ਸੰਿਖਆ",
#                                 "irrational numbers": "ਅਪਿਰਮਯ ਸੰਿਖਆ",
#                                 "real numbers": "ਵਾਸਤਿਵਕ ਸੰਿਖਆ",
#                                 "decimal expansions": "ਦ ਮਲਵ ਿਵਸਤਾਰ",
#                                 "terminating": "ਤ",
#                                 "non-terminating recurring": "ਅ ਤ ਆਵਰਤੀ",
#                                 "non-terminating non-recurring": "ਅ ਤ ਅਣ-ਆਵਰਤੀ",
#                                 "division method": "ਿਵਭਾਜਨ ਿਵਧੀ",
#                                 "number line": "ਸੰਿਖਆ ਰਖਾ",
#                                 "square root": "ਵਰਗ ਮਲ",
#                                 "positive": "ਧਨਾਤਮਕ",
#                                 "positive integers": "ਧਨਾਤਮਕ ਸੰਪਰਨ ਸੰਿਖਆਵ",
#                                 "rationalize": "ਪਿਰਮਯਕਰਨ",
#                                 "fractions": "ਿਭੰਨ",
#                                 "simlified fraction": "ਸਰਲ ਿਭੰਨ",
#                                 "equivalent expressions": "ਤਲ ਿਵਅੰਜਕ",
#                                 "rationaized form": "ਪਿਰਮਯਕਰਨ ਰਪ",
#                                 "polynomials": "ਬਹਪਦ"}
def correct_bhashni_translations(text,lowercase_dict):
    corrected_text = []
    words = text.split()  # Tokenize the input text into words
    for word in words:
        # Check if the word needs correction
        if word in lowercase_dict:
            corrected_text.append(lowercase_dict[word])
        else:
            corrected_text.append(word)  # If no correction needed, keep the word unchanged
    return ' '.join(corrected_text)

    #output_file=r"C:\Users\YH185MX\OneDrive - EY\Documents\dgr_punjab_project\education_pocs\inquizitive-main\question.pdf"
    # subprocess.run([
    #     "mdpdf", TEMP_MD_FILE,
    #     "--output", output_file,
    #     "--footer", ",,{page}",
    #     "--paper", "A4"
    # ])

    #os.remove(TEMP_MD_FILE)

#mode =  st.sidebar.selectbox('Choose demonstration mode', ['AI Interactive','Fixed Questions '])
#page = st.selectbox('Select Question Generation Mode', ['Question Generation Assistance'], index=0)
# if mode == 'Pre-loaded':
#     st.session_state.text, documents = chat.load_documents()
# elif mode == 'AI Interactive':
    
#     files = st.sidebar.file_uploader('Upload notes or lecture slides', accept_multiple_files=True, type=['pdf', 'docx', 'pptx', 'txt', 'md'])
#     learning_files = files if files is not None else []
#     st.session_state.text, documents = chat.load_documents(learning_files)

# files = st.file_uploader('Upload Documents', accept_multiple_files=True, type=['pdf', 'docx', 'pptx', 'txt', 'md'])
# learning_files = files if files is not None else []
# print("learning files are ")
# print(learning_files)

st.session_state.teach = st.radio("Select Option",(
    'Teachers','Students','Administration'),key='airadio1')
if st.session_state.teach=='Teachers':
    st.session_state.quesai = st.title("Generate Question and Answer")
    if st.session_state.quesai:
        #tab1, tab2,tab3= st.tabs(["1. Upload Document", "2. Text Analyzer","3. Skill based Questions"])
            choose=st.radio("Select Options",("Pre Uploaded","Text Analyzer","Skill Based Questions","Terminologies and Keyterms","Learning Outcomes"),horizontal=True)
            if choose=="Upload Documents":
                st.write('Note: File name should contain subject and class like maths_class10.pdf/.docx')
                files = st.file_uploader('Upload Books,Notes,Question Banks ', accept_multiple_files=True,type=['pdf', 'docx'])
                if files:
                    file_extension = files[0].name.split(".")[-1]
                    if file_extension == "pdf":
                        path = files[0].read()
                        name=files[0].name[:-4]
                        # Check if the file exists
                        if not os.path.exists("preuploaded/"+name+".txt"):
                            print("File Not Exist")
                            with open("preuploaded/"+name+'.txt', 'w',encoding='utf-8') as file:
                            # Open a file in write mode (creates a new file if it doesn't exist)
                                st.session_state.text= chat.load_pdf_text(files[0],name)
                                file.write(st.session_state.text)
                        else:
                            print("file Exist")
                            st.session_state.filename=[]
                            with open("preuploaded/"+name+'.txt', 'r',encoding='utf-8') as file:
                    #        Read the content of the file
                                st.session_state.filename.append(name)
                                st.session_state.text = file.read()
                        
                        # print("Extracted Text is ########################")
                        # print(st.session_state.text)
                        
                    elif file_extension =="docx":
                        st.session_state.filename=[]
                        doc_file=files[0]
                        doc_name=files[0].name[:-5]
                        st.session_state.filename.append(doc_name)
                        st.session_state.text= get_text(doc_file)
                        #st.write(st.session_state.text)
                        #print("Extracted Text is ########################")
                        #print(st.session_state.text)
                    else:
                        learning_files = files if files is not None else []
                        st.session_state.text, documents = chat.load_documents(learning_files)
                        # print("Extracted Text is ########################")
                        # print(st.session_state.text)
                        
                        # if st.session_state.text:
                        #         st.session_state.format_chain = ConversationChain( llm=AzureChatOpenAI(
                        #         deployment_name="gpt-4",
                        #         temperature=0
                        #         ))
                        #         formatted_output = chat.format_response(st.session_state.text)
                        #         st.info(formatted_output)
                    
                    if st.session_state.text:

                        # st.session_state.aiformat_chain = ConversationChain( llm=AzureChatOpenAI(
                        # deployment_name="gpt-4turbo",
                        # temperature=0
                        # ))
                        # formatted_output = chat.aiformat_response(st.session_state.text)
                        # st.info(formatted_output)
                        print("Inside Question Generation Bot")
                        with open('dictionary.json','r') as f:
                            existing_dictionary = json.load(f)

                        lowercase_dict = {key.lower(): value for key, value in existing_dictionary.items()}
                        
                        st.session_state.text=correct_bhashni_translations(st.session_state.text,lowercase_dict)
                        #st.write("Dictionary:", existing_dictionary)
                        # print(st.session_state.text)
                        # print(st.session_state.filename[0])
                        docsearch = chat.get_cache_vectorstore(st.session_state.text,st.session_state.filename[0])
                        print("embeddings Done")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.session_state.complexity =  st.selectbox('Complexity Mode Required?*', ['Easy', 'Difficult'],index=0,key="mode")
                            st.session_state.no_of_questions = st.number_input('No. of  Questions to generate*',key="ai_questions",step=1,max_value=30)
                            st.session_state.mode_of_questions = st.selectbox('Choose Answer Required/Not*', ['Only Questions', 'Questions with Answers'],index=0,key="quesansw")
                        with col2:
                            st.session_state.topic_name = st.text_input('Specific Chapter/Topic Name/Text*',placeholder="AI Chapter/Topic Name/Text")
                            st.session_state.type_of_questions =  st.selectbox('Choose Question Type*', ['Short Questions', 'Long Questions','MCQ','Fill in the Blanks','True and False'],index=0)
                            st.session_state.language =  st.selectbox('Choose Response Language Mode*', ['None','English','English and Hindi'],index=0,key="lang")
                            #docsearch = chat.create_doc_embeddings(documents)
                        
                    #if is_word_in_text(st.session_state.topic_name,formatted_output) or st.session_state.topic_name=='' :

                    
                    # Storing the chat
                    if 'generated' not in st.session_state:
                        st.session_state['generated'] = []

                    if 'past' not in st.session_state:
                        st.session_state['past'] = []

                    # Define a function to clear the input text
                    def clear_input_text():
                        global input_text
                        input_text = ""

                    # We will get the user's input by calling the get_text function
                    def get_text():
                        global input_text
                        input_text = st.chat_input("Ask a question!")
                        return input_text

                    user_input = get_text()

                    print('get the input text')

                    # Initialize chat history
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Display chat messages from history on app rerun
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            if message['content'] != user_input:
                                st.markdown(message["content"])

                    st.session_state.llm = load_chain(docsearch)
                    #print('chain loaded')
                    memory = ConversationBufferMemory(
                    return_messages=True,
                    )
                    st.session_state.language_chain = ConversationChain( llm=ChatOpenAI
                    (
                        model="gpt-3.5-turbo",
                        temperature=0.7
                        ),memory=memory)
                    _ = st.session_state.llm({'question':initialise_prompt})
                    if user_input:
                        english_output,trans_output = chat.answer(user_input)
                        print('get the output')
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        # Display user message in chat message container
                        with st.chat_message("user"):
                             st.markdown(user_input)  

                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            full_response = ""
                            for char in english_output:
                                full_response += char
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)

                            markdown_to_pdf(full_response,'question.pdf')
                            
                            word_doc = create_word_doc(full_response)
                            doc_buffer = download_doc(word_doc)
                            st.download_button(label="Download Word Document",
                                            data=doc_buffer, 
                                            file_name="generated_document.docx",
                                            mime="application/octet-stream",
                                            key='worddownload'
                                                )


                        st.session_state.messages.append({"role": "assistant", "content": english_output})

                        # Display assistant response in chat message container

                        if trans_output!="":
                            with st.chat_message("assistant"):
                                message_placeholder = st.empty()
                                full_response = ""
                                for char in trans_output:
                                    full_response += char
                                    message_placeholder.markdown(full_response + "▌")
                                message_placeholder.markdown(full_response)

                                markdown_to_pdf(full_response,'question.pdf')
                                
                                
                                word_doc = create_word_doc(full_response)
                                doc_buffer = download_doc(word_doc)
                                st.download_button(label="Download Word Document", 
                                                data=doc_buffer, 
                                                file_name="generated_document.docx", 
                                                mime="application/octet-stream",
                                                key='worddownload1')

                            st.session_state.messages.append({"role": "assistant", "content": full_response})

            
            
            
            if choose=="Pre Uploaded":
                def list_files(folder_path):
                    return os.listdir(folder_path)
                
                def remove_extension(filename):
                    return os.path.splitext(filename)[0]
                
                folder_path="./preuploaded"
                # Get list of files in folder
                files_list = list_files(folder_path)
                # Remove file extension from each filename
                files_list = [remove_extension(filename) for filename in files_list]
                files_list.insert(0, "Select document")
                # Display select box for selecting files
                selected_file = st.selectbox("Select a file", files_list)

                if selected_file!="Select document":
                    st.session_state.filename=[]
                    with open("preuploaded/"+selected_file+'.txt', 'r',encoding='ISO-8859-1') as file:
            #        Read the content of the file
                        st.session_state.filename.append(selected_file)
                        st.session_state.text = file.read()

                    if st.session_state.text:
                            # st.session_state.aiformat_chain = ConversationChain( llm=AzureChatOpenAI(
                            # deployment_name="gpt-4turbo",
                            # temperature=0
                            # ))
                            # formatted_output = chat.aiformat_response(st.session_state.text)
                            # st.info(formatted_output)
                            print("Inside Question Generation Bot")
                            
                            #st.write("Dictionary:", existing_dictionary)
                            # print(st.session_state.text)
                            # print(st.session_state.filename[0])
                            docsearch = chat.get_cache_vectorstore(st.session_state.text,st.session_state.filename[0])
                            print("embeddings Done")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.session_state.complexity =  st.selectbox('Complexity Mode Required?*', ['Easy', 'Difficult'],index=0,key="mode")
                                st.session_state.no_of_questions = st.number_input('No. of  Questions to generate*',key="ai_questions",step=1,max_value=30)
                                st.session_state.mode_of_questions = st.selectbox('Choose Answer Required/Not*', ['Only Questions', 'Questions with Answers'],index=0,key="quesansw")
                            with col2:
                                st.session_state.topic_name = st.text_input('Specific Chapter/Topic Name*',placeholder="AI Chapter/Topic Name")
                                st.session_state.type_of_questions =  st.selectbox('Choose Question Type*', ['Short Questions', 'Long Questions','MCQ','Fill in the Blanks','True and False'],index=0)
                                st.session_state.language =  st.selectbox('Choose Response Language Mode*', ['English'],index=0,key="lang")
                                #docsearch = chat.create_doc_embeddings(documents)
                            
                        #if is_word_in_text(st.session_state.topic_name,formatted_output) or st.session_state.topic_name=='' :

                        
                    # Storing the chat
                    if 'generated' not in st.session_state:
                        st.session_state['generated'] = []

                    if 'past' not in st.session_state:
                        st.session_state['past'] = []

                    # Define a function to clear the input text
                    def clear_input_text():
                        global input_text
                        input_text = ""

                    # We will get the user's input by calling the get_text function
                    def get_text():
                        global input_text
                        input_text = st.chat_input("Ask a question!")
                        return input_text

                    user_input = get_text()

                    print('get the input text')

                    # Initialize chat history
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Display chat messages from history on app rerun
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            if message['content'] != user_input:
                                st.markdown(message["content"])

                    st.session_state.llm = load_chain(docsearch)
                    print('chain loaded')
                    memory = ConversationBufferMemory(return_messages=True)
                            
                    st.session_state.language_chain = ConversationChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.7),memory=memory) 
                    _= st.session_state.llm({'question':initialise_prompt})
                    
                    if user_input:
                        english_output,trans_output = chat.answer(user_input)

                        print('get the output')
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        # Display user message in chat message container
                        with st.chat_message("user"):
                            st.markdown(user_input)  

                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            full_response = ""
                            for char in english_output:
                                full_response += char
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)

                            markdown_to_pdf(full_response,'question.pdf')
                            
                            
                            word_doc = create_word_doc(full_response)
                            doc_buffer = download_doc(word_doc)
                            st.download_button(label="Download Word Document",
                                            data=doc_buffer, 
                                            file_name="generated_document.docx",
                                            mime="application/octet-stream",
                                            key='worddownload'
                                                )
                        st.session_state.messages.append({"role": "assistant", "content": english_output})

                        # Display assistant response in chat message container

                        


            
            if choose=="Terminologies and Keyterms":
                def list_files(folder_path):
                    return os.listdir(folder_path)
                
                def remove_extension(filename):
                    return os.path.splitext(filename)[0]
                
                folder_path="./preuploaded"
                # Get list of files in folder
                files_list = list_files(folder_path)
                # Remove file extension from each filename
                files_list = [remove_extension(filename) for filename in files_list]
                files_list.insert(0, "Select document")
                # Display select box for selecting files
                selected_file = st.selectbox("Select a file", files_list)

                if selected_file!="Select document":
                    st.session_state.filename=[]
                    with open("preuploaded/"+selected_file+'.txt', 'r',encoding='ISO-8859-1') as file:
            #        Read the content of the file
                        st.session_state.filename.append(selected_file)
                        text = file.read()
                    #     Read the content of the file
                        st.session_state.mcq_chain = ConversationChain( llm=ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.7
                        ))
                        outputs = chat.mcq_response(text)
                        #st.write(st.session_state.outputs)
                        #st.write(outputs)
                        markdown_to_pdf(outputs,'question.pdf')
                        
                        
                        word_doc = create_word_doc(outputs)
                        doc_buffer = download_doc(word_doc)
                        st.download_button(label="Download Word Document", 
                                        data=doc_buffer, 
                                        file_name="generated_document.docx", 
                                        mime="application/octet-stream",
                                        key='worddownload3')
            
            

            if choose=="Learning Outcomes":
                def list_files(folder_path):
                    return os.listdir(folder_path)
                
                def remove_extension(filename):
                    return os.path.splitext(filename)[0]
                
                folder_path="./preuploaded"
                # Get list of files in folder
                files_list = list_files(folder_path)
                # Remove file extension from each filename
                files_list = [remove_extension(filename) for filename in files_list]
                files_list.insert(0, "Select document")
                # Display select box for selecting files
                selected_file = st.selectbox("Select a file", files_list)

                if selected_file!="Select document":
                    st.session_state.filename=[]
                    with open("preuploaded/"+selected_file+'.txt', 'r',encoding='ISO-8859-1') as file:
            #        Read the content of the file
                        st.session_state.filename.append(selected_file)
                        text = file.read()
                        st.session_state.learn_outcome_chain = ConversationChain(llm=ChatOpenAI(
                        model = "gpt-3.5-turbo",
                        temperature=0.7
                        ))
                        outputs = chat.learn_outcome_term(text)
                        #st.write(outputs)
                        markdown_to_pdf(outputs,'question.pdf')
                        
                        
                        word_doc = create_word_doc(outputs)
                        doc_buffer = download_doc(word_doc)
                        st.download_button(label="Download Word Document", 
                                        data=doc_buffer, 
                                        file_name="generated_document.docx", 
                                        mime="application/octet-stream",
                                        key='worddownload3')


            if choose=="Text Analyzer":
                txt = st.text_area(
                "Text to Generate Questions"
                )
                if len(txt)>0:
                    #st.write(txt)
                    with open('dictionary.json','r') as f:
                            existing_dictionary = json.load(f)

                    lowercase_dict = {key.lower(): value for key, value in existing_dictionary.items()}
                    st.session_state.text=correct_bhashni_translations(txt,lowercase_dict)
                    st.write(st.session_state.text)

                    col1, col2 = st.columns(2)
                    with col1:
                        #st.session_state.complexity =  st.selectbox('Complexity Mode Required?*', ['No', 'Yes'],index=0,key="mode")
                        st.session_state.no_of_questions = st.number_input('No. of  Questions to generate*',key="ai_questions_no_k",step=1,max_value=30)
                        st.session_state.mode_of_questions = st.selectbox('Choose Answer Required/Not*', ['Select Option','Only Questions', 'Questions with Answers'],index=0,key="quesansw")
                    with col2:
                        st.session_state.type_of_questions =  st.selectbox('Choose Question Type*', ['Select Option','Short Questions','MCQ','Fill in the Blanks','True and False'],index=0,key="ai_questions_no_s")
                    if st.session_state.text and st.session_state.mode_of_questions!='Select Option' :
                            st.session_state.llm = ConversationChain( llm=ChatOpenAI(
                            model="gpt-3.5-turbo",
                            temperature=0.7
                            )) 
                            formatted_output = st.session_state.llm.predict(input = ai_prompt.format(st.session_state.no_of_questions,
                                                                            st.session_state.mode_of_questions,
                                                                            st.session_state.type_of_questions,
                                                                            st.session_state.text))
                            st.info(formatted_output)
                            markdown_to_pdf(formatted_output,'question.pdf')
                           
                           
                            word_doc = create_word_doc(formatted_output)
                            doc_buffer = download_doc(word_doc)
                            st.download_button(label="Download Word Document", 
                                            data=doc_buffer, 
                                            file_name="generated_document.docx", 
                                            mime="application/octet-stream",
                                            key='worddownload2')

            else:
                    st.write("")

            
   
                
            if choose=="Skill Based Questions":
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.topic_name = st.text_input('Specific Topic Name',placeholder="Topic Name",key="tt")
                    st.session_state.complexity =  st.selectbox('Complexity Mode Required?*', ['Easy', 'Difficult'],index=0,key="mode1")
                    st.session_state.no_of_questions = st.number_input('No. of  Questions to generate*',key="ai_questions_no_a",step=1,max_value=30)
                with col2:
                    st.session_state.type_of_questions =  st.selectbox('Choose Question Type*', ['Select Option','Short Questions','MCQ','Fill in the Blanks','True and False'],index=0,key="ai_questions_no_p")
                    st.session_state.classq =  st.selectbox('Choose Class*', ['Select Option','1','2','3','4','5','6','7','8','9','10','11','12'],index=0,key="ai_questions_no_p1")
                    st.session_state.mode_of_questions = st.selectbox('Choose Answer Required/Not*', ['Select Option','Only Questions', 'Questions with Answers'],index=0,key="quesanswz")
                if  st.session_state.topic_name and st.session_state.mode_of_questions!='Select Option' :
                    st.session_state.llm = ConversationChain( llm=ChatOpenAI(
                                                              model = "gpt-3.5-turbo",
                                                              temperature=0.7
                                                              ))

                    formatted_output = st.session_state.llm.predict(input = ai_topic_prompt.format(st.session_state.topic_name,
                                                                                                st.session_state.no_of_questions,
                                                                                                    st.session_state.mode_of_questions,
                                                                                                    st.session_state.type_of_questions,
                                                                                                    st.session_state.classq,
                                                                                                  st.session_state.complexity))
                    
                    # formatted_output = formatted_output.replace("^", r"^{")  # Replace ^ with ^
                    # formatted_output = formatted_output.replace("x", r"x")  # Replace x with x
                    # formatted_output = formatted_output.replace("?", r"?")  # Replace ? with ?
                    # formatted_output = formatted_output.replace("A)", r"A) ")  # Add space after A)
                    # formatted_output = formatted_output.replace("B)", r"B) ")  # Add space after B)
                    # formatted_output = formatted_output.replace("C)", r"C) ")  # Add space after C)
                    # formatted_output = formatted_output.replace("D)", r"D) ")  # Add space after D)
                    # formatted_output = formatted_output.replace("(", r"\left(")  # Replace ( with \left(
                    # formatted_output = formatted_output.replace(")", r"\right)")  # Replace ) with \right)
                    # questions = text_to_latex(formatted_output)
                    # formatted_text = add_dollar_signs(questions)
                    #st.write(formatted_text)
                    
                    #latex_formatted_output=st.session_state.llm.predict(input = latex_prompt.format(formatted_output))
                    #questions = re.findall(r'"(.*?)"',latex_formatted_output)
                    st.write(formatted_output)                                                 
                    markdown_to_pdf(formatted_output,'question.pdf')
                    
                    
                    word_doc = create_word_doc(formatted_output)
                    doc_buffer = download_doc(word_doc)
                    st.download_button(label="Download Word Document", 
                                    data=doc_buffer, 
                                    file_name="generated_document.docx", 
                                    mime="application/octet-stream",
                                    key='worddownload3')
                    
if st.session_state.teach=='Students':
    choose=st.radio("Select Options",("Ask a Query","Text Analyzer"),horizontal=True)

    if choose=="Text Analyzer":
                
                txt = st.text_area(
                "Text to Generate Questions"
                )
                if len(txt)>0:
                    #st.write(txt)
                    with open('dictionary.json','r') as f:
                            existing_dictionary = json.load(f)

                    lowercase_dict = {key.lower(): value for key, value in existing_dictionary.items()}
                    st.session_state.text=correct_bhashni_translations(txt,lowercase_dict)
                    st.write(st.session_state.text)

                    col1, col2 = st.columns(2)
                    with col1:
                        #st.session_state.complexity =  st.selectbox('Complexity Mode Required?*', ['No', 'Yes'],index=0,key="mode")
                        st.session_state.no_of_questions = st.number_input('No. of  Questions to generate*',key="ai_questions_no_ks",step=1,max_value=30)
                        st.session_state.mode_of_questions = st.selectbox('Choose Answer Required/Not*', ['Select Option','Only Questions', 'Questions with Answers'],index=0,key="quesansws")
                    with col2:
                        st.session_state.type_of_questions =  st.selectbox('Choose Question Type*', ['Select Option','Short Questions','MCQ','Fill in the Blanks','True and False'],index=0,key="ai_questions_no_ss")
                    if st.session_state.text and st.session_state.mode_of_questions!='Select Option' :
                            st.session_state.llm = ConversationChain( llm=ChatOpenAI(
                            model_name="gpt-3.5-turbo",
                            temperature=0.7
                            )) 
                            formatted_output = st.session_state.llm.predict(input = ai_prompt.format(st.session_state.no_of_questions,
                                                                            st.session_state.mode_of_questions,
                                                                            st.session_state.type_of_questions,
                                                                            st.session_state.text))
                            st.info(formatted_output)
                            markdown_to_pdf(formatted_output,'question.pdf')
                            
                            word_doc = create_word_doc(formatted_output)
                            doc_buffer = download_doc(word_doc)
                            st.download_button(label="Download Word Document", 
                                            data=doc_buffer, 
                                            file_name="generated_document.docx", 
                                            mime="application/octet-stream",
                                            key='worddownload2')

    else:
            st.write("")
    if choose=="MCQ Test":
        st.write("In Development Stage")
        st.write('Note: File name should contain subject and class like maths_class10.pdf/.docx')
        files = st.file_uploader('Upload Books,Notes,Question Banks ', accept_multiple_files=True,type=['pdf', 'docx'])
        if files:
            file_extension = files[0].name.split(".")[-1]
            if file_extension == "pdf":
                path = files[0].read()
                name=files[0].name[:-4]
                # Check if the file exists
                if not os.path.exists(name+".txt"):
                    print("File Not Exist")
                    with open(name+'.txt', 'w') as file:
                    # Open a file in write mode (creates a new file if it doesn't exist)
                        st.session_state.text= chat.load_pdf_text(files[0],name)
                        file.write(st.session_state.text)
                        # st.session_state.mcq_chain = ConversationChain( llm=AzureChatOpenAI(
                        # deployment_name="gpt-4turbo",
                        # temperature=0
                        # ))
                        #outputs = chat.mcq_response(st.session_state.text)
                        outputs = generate_quiz(st.session_state.text)
                        #st.info(formatted_output)
                        try:
                            quiz_json = json.loads(outputs[0])["quiz"]
                            st.session_state['quiz_length'] = len(quiz_json)
                            questions = [q['question'] for q in quiz_json]
                            options = [q['options'] for q in quiz_json]
                            answers = [q["answer"] for q in quiz_json]
                            explanations = [q['explanation'] for q in quiz_json]
                            
                            # user selects which question they want to answer
                            # question_num = st.number_input('Choose a question', min_value=1, max_value = len(quiz_json), value=1)
                            question_num = st.session_state['curr_question']
                            answer_choices = options[question_num]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.button('Previous Question', use_container_width=True, on_click=decrement_question_num)
                            with col2:
                                st.button('Next Question', use_container_width=True, on_click=increment_question_num)

                            st.markdown(f"##### Question {question_num + 1} of {len(quiz_json)}: {questions[question_num]}")

                            if 'a' not in st.session_state:
                                st.session_state.a = 0
                                st.session_state.b = 0
                                st.session_state.c = 0
                                st.session_state.d = 0

                            def ChangeA():
                                st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 1,0,0,0
                            def ChangeB():
                                st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 0,1,0,0
                            def ChangeC():
                                st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 0,0,1,0
                            def ChangeD():
                                st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 0,0,0,1
                            def ClearAll():
                                st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 0,0,0,0

                            checkboxA = st.checkbox(answer_choices[0], value = st.session_state.a, on_change = ChangeA)
                            checkboxB = st.checkbox(answer_choices[1], value = st.session_state.b, on_change = ChangeB)
                            checkboxC = st.checkbox(answer_choices[2], value = st.session_state.c, on_change = ChangeC)
                            checkboxD = st.checkbox(answer_choices[3], value = st.session_state.d, on_change = ChangeD)

                            if st.session_state.a:
                                user_answer = answer_choices[0]
                            elif st.session_state.b:
                                user_answer = answer_choices[1]
                            elif st.session_state.c:
                                user_answer = answer_choices[2]
                            elif st.session_state.d:
                                user_answer = answer_choices[3]
                            else:
                                user_answer = None

                            if user_answer is not None:
                                user_answer_num = answer_choices.index(user_answer)
                                if st.button('Submit Answer', type='secondary'):
                                    if user_answer_num == answers[question_num][0]:
                                        st.success(f'Correct! {explanations[question_num]}')
                                    else:
                                        st.error(f'Incorrect :( \n\n The correct answer was: {answer_choices[answers[question_num][0]]}\n\n {explanations[question_num]}')
                        except:
                            st.info('Uh oh... could not generate a quiz for ya! Happy studying!')
                else:
                    print("file Exist")
                    st.session_state.filename=[]
                    with open(name+'.txt', 'r',encoding='ISO-8859-1') as file:
            #        Read the content of the file
                        st.session_state.filename.append(name)
                        st.session_state.text = file.read()
                        #st.write(st.session_state.text)
                        #outputs = generate_quiz(st.session_state.text)
                        # st.session_state.mcq_chain = ConversationChain( llm=AzureChatOpenAI(
                        # deployment_name="gpt-4turbo",
                        # temperature=0
                        # ))
                        #outputs = chat.mcq_response(st.session_state.text)
                        outputs = generate_quiz(st.session_state.text)
                        #st.write(outputs)
                        #print("Output are")
                        #print(outputs)
                        
                        st.write(outputs)
                        corrected_json_data = outputs[0].replace('//', '')
                        quiz_json = json.loads(corrected_json_data)["quiz"]
                        st.session_state['quiz_length'] = len(quiz_json)
                        questions = [q['question'] for q in quiz_json]
                        options = [q['options'] for q in quiz_json]
                        answers = [q["answer"] for q in quiz_json]
                        explanations = [q['explanation'] for q in quiz_json]
                        
                        # user selects which question they want to answer
                        # question_num = st.number_input('Choose a question', min_value=1, max_value = len(quiz_json), value=1)
                        question_num = st.session_state['curr_question']
                        answer_choices = options[question_num]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.button('Previous Question', use_container_width=True, on_click=decrement_question_num)
                        with col2:
                            st.button('Next Question', use_container_width=True, on_click=increment_question_num)

                        st.markdown(f"##### Question {question_num + 1} of {len(quiz_json)}: {questions[question_num]}")

                        if 'a' not in st.session_state:
                            st.session_state.a = 0
                            st.session_state.b = 0
                            st.session_state.c = 0
                            st.session_state.d = 0

                        def ChangeA():
                            st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 1,0,0,0
                        def ChangeB():
                            st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 0,1,0,0
                        def ChangeC():
                            st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 0,0,1,0
                        def ChangeD():
                            st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 0,0,0,1
                        def ClearAll():
                            st.session_state.a,st.session_state.b,st.session_state.c,st.session_state.d = 0,0,0,0

                        checkboxA = st.checkbox(answer_choices[0], value = st.session_state.a, on_change = ChangeA)
                        checkboxB = st.checkbox(answer_choices[1], value = st.session_state.b, on_change = ChangeB)
                        checkboxC = st.checkbox(answer_choices[2], value = st.session_state.c, on_change = ChangeC)
                        checkboxD = st.checkbox(answer_choices[3], value = st.session_state.d, on_change = ChangeD)

                        if st.session_state.a:
                            user_answer = answer_choices[0]
                        elif st.session_state.b:
                            user_answer = answer_choices[1]
                        elif st.session_state.c:
                            user_answer = answer_choices[2]
                        elif st.session_state.d:
                            user_answer = answer_choices[3]
                        else:
                            user_answer = None

                        if user_answer is not None:
                            user_answer_num = answer_choices.index(user_answer)
                            if st.button('Submit Answer', type='secondary'):
                                if user_answer_num == answers[question_num][0]:
                                    st.success(f'Correct! {explanations[question_num]}')
                                else:
                                    st.error(f'Incorrect :( \n\n The correct answer was: {answer_choices[answers[question_num][0]]}\n\n {explanations[question_num]}')

    if choose=="Ask a Query":
                
                files = st.file_uploader('Upload Books,Notes,Question Banks ', accept_multiple_files=True,type=['pdf', 'docx'])
                if files:
                    file_extension = files[0].name.split(".")[-1]
                    if file_extension == "pdf":
                        path = files[0].read()
                        name=files[0].name[:-4]
                        # Check if the file exists
                        if not os.path.exists("studentuploaded/"+name+".txt"):
                            print("File Not Exist")
                            with open("studentuploaded/"+name+'.txt', 'w',encoding='utf-8') as file:
                            # Open a file in write mode (creates a new file if it doesn't exist)
                                st.session_state.text= chat.load_pdf_text(files[0],name)
                                file.write(st.session_state.text)
                        else:
                            print("file Exist")
                            st.session_state.filename=[]
                            with open("studentuploaded/"+name+'.txt', 'r',encoding='utf-8') as file:
                    #        Read the content of the file
                                st.session_state.filename.append(name)
                                st.session_state.text = file.read()
                        
                        # print("Extracted Text is ########################")
                        # print(st.session_state.text)
                        
                    elif file_extension =="docx":
                        st.session_state.filename=[]
                        doc_file=files[0]
                        doc_name=files[0].name[:-5]
                        st.session_state.filename.append(doc_name)
                        st.session_state.text= get_text(doc_file)
                        #st.write(st.session_state.text)
                        #print("Extracted Text is ########################")
                        #print(st.session_state.text)
                    else:
                        learning_files = files if files is not None else []
                        st.session_state.text, documents = chat.load_documents(learning_files)
                        # print("Extracted Text is ########################")
                        # print(st.session_state.text)
                        
                        # if st.session_state.text:
                        #         st.session_state.format_chain = ConversationChain( llm=AzureChatOpenAI(
                        #         deployment_name="gpt-4",
                        #         temperature=0
                        #         ))
                        #         formatted_output = chat.format_response(st.session_state.text)
                        #         st.info(formatted_output)
                    
                    if st.session_state.text:

                        # st.session_state.aiformat_chain = ConversationChain( llm=AzureChatOpenAI(
                        # deployment_name="gpt-4turbo",
                        # temperature=0
                        # ))
                        # formatted_output = chat.aiformat_response(st.session_state.text)
                        # st.info(formatted_output)
                        print("Inside Question Generation Bot")
                        with open('dictionary.json','r') as f:
                            existing_dictionary = json.load(f)

                        lowercase_dict = {key.lower(): value for key, value in existing_dictionary.items()}

                        docsearch = chat.get_cache_vectorstore(st.session_state.text,st.session_state.filename[0])
                        # Define a function to clear the input text
                        # Storing the chat
                    if 'generated' not in st.session_state:
                        st.session_state['generated'] = []

                    if 'past' not in st.session_state:
                        st.session_state['past'] = []

                    # Define a function to clear the input text
                    def clear_input_text():
                        global input_text
                        input_text = ""

                    # We will get the user's input by calling the get_text function
                    def get_text():
                        global input_text
                        input_text = st.chat_input("Ask a question!")
                        return input_text

                    user_input = get_text()

                    print('get the input text')

                    # Initialize chat history
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Display chat messages from history on app rerun
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            if message['content'] != user_input:
                                st.markdown(message["content"])
                    
                    st.session_state.llm = load_chain(docsearch)
                    print('chain loaded')
                    memory = ConversationBufferMemory(
                    return_messages=True,
                    )
                    st.session_state.language_chain = ConversationChain( llm=ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.7
                        ),memory=memory)
                    _ = st.session_state.llm({'question':initialise_prompt})
                    if user_input:
                        english_output = chat.answerq(user_input,st.session_state.text)
                        print('get the output')
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        # Display user message in chat message container
                        with st.chat_message("user"):
                            st.markdown(user_input)  

                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            full_response = ""
                            for char in english_output:
                                full_response += char
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)

                            markdown_to_pdf(full_response,'question.pdf')
                           
                            word_doc = create_word_doc(full_response)
                            doc_buffer = download_doc(word_doc)
                            st.download_button(label="Download Word Document",
                                            data=doc_buffer, 
                                            file_name="generated_document.docx",
                                            mime="application/octet-stream",
                                            key='worddownload'
                                                )


                        st.session_state.messages.append({"role": "assistant", "content": english_output})
    if choose=="Terminologies and Keyterms":
        st.write('Note: File name should contain subject and class like maths_class10.pdf/.docx')
        files = st.file_uploader('Upload Books,Notes,Question Banks ', accept_multiple_files=True,type=['pdf', 'docx'])
        if files:
            file_extension = files[0].name.split(".")[-1]
            if file_extension == "pdf":
                path = files[0].read()
                name=files[0].name[:-4]
                #Check if the file exists
                if not os.path.exists(name+".txt"):
                    print("File Not Exist")
                    with open(name+'.txt', 'w') as file:
                    #Open a file in write mode (creates a new file if it doesn't exist)
                        st.session_state.text= chat.load_pdf_text(files[0],name)
                        file.write(st.session_state.text)
                else:
                    print("file Exist")
                    st.session_state.filename=[]
                    with open(name+'.txt', 'r',encoding='ISO-8859-1') as file:
                    #Read the content of the file
                        st.session_state.filename.append(name)
                        st.session_state.text = file.read()
                        st.session_state.mcq_chain = ConversationChain( llm=ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.7
                        ))
                        outputs = chat.mcq_response(st.session_state.text)
                        st.write(outputs)
                        markdown_to_pdf(outputs,'question.pdf')
                        
                        word_doc = create_word_doc(outputs)
                        doc_buffer = download_doc(word_doc)
                        st.download_button(label="Download Word Document", 
                                        data=doc_buffer, 
                                        file_name="generated_document.docx", 
                                        mime="application/octet-stream",
                                        key='worddownload3')


if st.session_state.teach=='Administration':
    choose=st.radio("Select Options",("Add Document","Download Document","Delete Document","View Documents"),horizontal=True)

    if choose=="Add Document":
        files = st.file_uploader('Upload Books,Notes,Question Banks ', accept_multiple_files=True,type=['pdf', 'docx'])
        if files:
            file_extension = files[0].name.split(".")[-1]
            if file_extension == "pdf":
                path = files[0].read()
                name=files[0].name[:-4]
                # Check if the file exists
                if not os.path.exists("preuploaded/"+name+".txt"):
                    print("File Not Exist")
                    with open("preuploaded/"+name+'.txt', 'w',encoding='utf-8') as file:
                    # Open a file in write mode (creates a new file if it doesn't exist)
                        st.session_state.text= chat.load_pdf_text(files[0],name)
                        file.write(st.session_state.text)
                else:
                    st.write("Document Exist")

    if choose=="Download Document":
        # Example usage
        folder_path = "./preuploaded"

        files = os.listdir(folder_path)
                    
        # Filter out only text files
        text_files = [file for file in files if file.endswith(".txt")]

        text_files.insert(0, "Select document")

        # Create a selectbox with multi-selection enabled
        selected_file = st.selectbox("Select Document", text_files)

        if selected_file != "Select document":
            # Display download button for the selected file
            st.write("Download selected file:")
            with open(os.path.join(folder_path, selected_file), "rb") as file:
                st.download_button(label="Download", data=file, file_name=selected_file, mime="text/plain")
        else:
            st.write("Select Document to Download")


    if choose=="View Documents":
        def view_text_files(folder_path):
            # Get a list of all files in the folder
            files = os.listdir(folder_path)
            
            # Filter out only text files
            text_files = [file for file in files if file.endswith(".txt")]

            if text_files:
                st.write("Documents in the folder:")
                for text_file in text_files:
                    st.write(text_file)
            else:
                st.write("No Documents found in the folder.")

        # Example usage
        folder_path = "./preuploaded"
        view_text_files(folder_path)

    if choose=="Delete Document":
        def remove_text_file(folder_path, file_name):
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                st.write(f"{file_name} has been successfully removed")
            else:
                st.write(f"File {file_name} does not exist in {folder_path}")

        # Example usage
        folder_path = "./preuploaded"

        files = os.listdir(folder_path)
            
            # Filter out only text files
        text_files = [file for file in files if file.endswith(".txt")]

        text_files.insert(0, "Select document")

        # Create a selectbox with multi-selection enabled
        selected_file = st.selectbox("Select Document", text_files)

        # Display the selected files
        if selected_file!="Select document":
            remove_text_file(folder_path, selected_file)
        else:
            st.write("Select Document to Delete")
        
        
    if choose=="Add Word to Dictionary":
        if st.button("View Dictionary"):
            with open('dictionary.json','r') as f:
                    
                    existing_dictionary = json.load(f)

            lowercase_dict = {value.lower(): key for key, value in existing_dictionary.items()}
            st.write(lowercase_dict)

        # Input field for searching a word
        search_word = st.text_input('Enter English Word to Search:', '')
        if search_word:
            with open('dictionary.json','r') as f:
                existing_dictionary = json.load(f)
            translation = existing_dictionary.get(search_word, 'Word not found')
            st.write(f"Hindi Translation for '{search_word}': {translation}")

        with open('dictionary.json','r') as f:
            st.session_state.existing_dictionary = json.load(f)

        # Function to save dictionary to a local file
        def save_dictionary_to_file(filename):
            #st.write(st.session_state.existing_dictionary)
            with open(filename, 'w') as f:
                json.dump(st.session_state.existing_dictionary, f)

        col1, col2 = st.columns(2)
        with col1:

            st.write("##### English Words (Separated by commas)")
            eng_txt = st.text_input(
                "Enter English Words",
                key="eng"
            )
        with col2:

            st.write("##### Hindi Words (Separated by commas)")
            punjabi_txt = st.text_input(
                "Enter Hindi Words",
                key="hi"
            )

        lowercase_dict = {key.lower(): value for key, value in st.session_state.existing_dictionary.items()}

        if st.button("Request to Add"):
        # Split the entered English and Punjabi words by commas and remove leading/trailing whitespace
            eng_words = [word.strip() for word in eng_txt.split(",")]
            punjabi_words = [word.strip() for word in punjabi_txt.split(",")]

            # Add each pair of words to the lowercase dictionary
            for eng_word, punjabi_word in zip(eng_words, punjabi_words):
                # Convert entered English word to lowercase
                eng_lower = eng_word.lower()
                # Add the entry to the lowercase dictionary
                st.session_state.existing_dictionary[eng_lower] = punjabi_word
            st.success("Words added to dictionary!")
            st.write("Dictionary:", st.session_state.existing_dictionary)
            save_dictionary_to_file("dictionary.json")
            st.success("Dictionary saved to local!")

        






                # if formatted_output and st.session_state.quesai=='Yes':
                #     # if not hasattr(st.session_state,"llm"):
                #     st.session_state.llm = ConversationChain( llm=AzureChatOpenAI(
                #     deployment_name="gpt-4",
                #     temperature=0
                #     ))  
                #     st.session_state.topic_name = st.text_input('Enter Topic Name for which you want to generate questions',placeholder="Topic Name",key="aitopic")
                #     if st.session_state.topic_name:
                #         st.session_state.type_of_questions =  st.selectbox('Choose Question Type*',['Select Option','Multiple Choice Questions','True/False'],index=0,key="aiqtype")
                #         if st.session_state.type_of_questions!='Select Option':
                #             st.session_state.no_of_questions = st.number_input('No. of  Questions*',step=1,key="ainoques")
                #             print(type(st.session_state.no_of_questions))
                #             if st.session_state.type_of_questions:

                #                 result = st.session_state.llm.predict(input = ai_prompt.format(st.session_state.no_of_questions,
                #                                                     st.session_state.type_of_questions,
                #                                                     st.session_state.topic_name,
                #                                                     st.session_state.text))
                #                 st.session_state.llm.memory.clear()
                #                 st.markdown("#### AI Generated Questions")
                #                 st.success(result)
                #                 markdown_to_pdf(result,'question.pdf')
                #                 st.download_button(
                #                 "Download",
                #                 data=open("question.pdf", "rb").read(),
                #                 file_name="question.pdf",
                #                 mime="application/pdf",
                #                 help="Download the questions as a PDF file"
                #                 )



                        #st.session_state.quesai = st.radio("Do you want AI Generated Questions Required?",('No','Yes'))



        # st.session_state.fig =  st.selectbox('Figure based Questions Required*', ['No','Yes'],index=0)
        # col1, col2 = st.columns(2)
        # if st.session_state.fig=='Yes':
        #     with col1:
        #         st.session_state.subject =  st.selectbox('Choose Subject*', ['Select Subect','Maths', 'English','Social Science'],index=0)
        #         #st.write(st.write('You selected:',  st.session_state.class_sub))
        #         st.session_state.topic_name = st.selectbox('Choose Topic Name*', list(ques['Topic'].unique()),index=0)
        #         #st.session_state.complexity =  st.selectbox('Choose complexity mode*', ['Easy', 'Medium','Complex'],index=0,key="mode")
        #         st.session_state.no_of_questions = st.number_input('No. of  Questions*',key="fix_questions",step=1)
        #         # st.session_state.mode_of_questions = st.selectbox('Choose Answer Required/Not*', ['Only Questions', 'Questions with Answers'],index=0,key="quesansw")
        #         #st.session_state.ques = st.radio("Fixed Questions Required?",('Yes','No'))
        #     with col2:
        #             st.session_state.class_sub =  st.selectbox('Choose Class*', list(ques['Class'].unique()),index=0)
        #             st.session_state.type_of_questions =  st.selectbox('Choose Question Type*', list(ques['Type of Question'].unique()),index=0)
        #             st.session_state.ques = st.radio("AI Generated Questions Required?",('No','Yes'))

        #     if st.session_state.topic_name is not None:
        #     # text = get_text(uploaded_file)
        #         st.session_state.format_chain = ConversationChain( llm=AzureChatOpenAI(
        #         deployment_name="gpt-4",
        #         temperature=0
        #         ))

        #         # if text:
        #         #     formatted_output = chat.format_response(text)
        #         # Define the path to your JSON file
        #         file_path = st.session_state.topic_name+".json"

        #         # Open the JSON file and load its contents
        #         # with open(file_path, "r",encoding="utf-8") as file:
        #             # data = json.load(file)
        #         data=ques
        #             #print(data)

        #         st.markdown("#### Fixed Questions")
        #         try:

        #             # fix_result=data[st.session_state.topic_name][st.session_state.type_of_questions]
        #             print(st.session_state.topic_name,st.session_state.type_of_questions)
        #             fix_result=data[(data['Topic']==st.session_state.topic_name) & (data['Type of Question']==st.session_state.type_of_questions)].sample(frac=1).reset_index(drop=True)[:int(st.session_state.no_of_questions)]
        #             # fix_result = 
                    
        #             print(fix_result)
                    
        #             try:
        #                 # formatted_output =display_mcq(fix_result)
        #                 display_q(fix_result)
        #             except Exception as e:
        #                 print(e)

        #         except Exception as e:
        #             print(e)
        #             st.info("Questions not available,Refer to AI Generated Questions Options")


        # if st.session_state.fig=='No': 
        #     uploaded_file = st.file_uploader("Upload a .docx file", type="docx")
        #     with col1:
        #         st.session_state.topic_name = st.text_input('Specific Chapter/Topic Name(Optional)',placeholder="Topic Name")
        #         #st.session_state.complexity =  st.selectbox('Choose complexity mode*', ['Easy', 'Medium','Complex'],index=0,key="mode")
        #         st.session_state.no_of_questions = st.number_input('No. of  Questions*',key="fix_questions",step=1)
        #         # st.session_state.mode_of_questions = st.selectbox('Choose Answer Required/Not*', ['Only Questions', 'Questions with Answers'],index=0,key="quesansw")
        #         #st.session_state.ques = st.radio("Fixed Questions Required?",('Yes','No'))
        #     with col2:
        #             st.session_state.type_of_questions =  st.selectbox('Choose Question Type*', list(ques['Type of Question'].unique()),index=0)
        #             st.session_state.ques = st.radio("AI Generated Questions Required?",('No','Yes'))
        #     text = get_text(uploaded_file)
        #     if text:
        #             st.session_state.format_chain = ConversationChain( llm=AzureChatOpenAI(
        #             deployment_name="gpt-4",
        #             temperature=0
        #             ))
        #             formatted_output = chat.format_response(text)
        #             st.info(formatted_output)

        #             json_data = extract_questions(text)
        #             with open("output.json", "w", encoding="utf-8") as f:
        #                 json.dump(json_data, f, indent=4, ensure_ascii=False)

            #st.session_state.language =  st.selectbox('Choose Response Language Mode*', ['None','Raavi Punjabi','Hindi'],index=0,key="lang")

        # if st.session_state.quesai == 'Yes' and st.session_state.topic_name is not None:
        #     if not hasattr(st.session_state,"llm"):
        #         st.session_state.llm = ConversationChain( llm=AzureChatOpenAI(
        #         deployment_name="gpt-4",
        #         temperature=0
        #         ))  
        #     result = st.session_state.llm.predict(input = ai_prompt.format(st.session_state.no_of_questions,
        #                                         st.session_state.type_of_questions,
        #                                         st.session_state.topic_name,
        #                                         st.session_state.class_sub,
        #                                         text))
        #     st.session_state.llm.memory.clear()
        #     st.markdown("#### AI Generated Questions")
        #     st.success(result)
        #     markdown_to_pdf(result,'question.pdf')
        #     st.download_button(
        #     "Download",
        #     data=open("question.pdf", "rb").read(),
        #     file_name="question.pdf",
        #     mime="application/pdf",
        #     help="Download the questions as a PDF file"
        #     )

                
    
#@st.cache_data()





# if st.sidebar.button('Process Study Materials'):
#     st.session_state['materials_processed'] = True
#     st.session_state['curr_question'] = 0
   


# if 'materials_processed' in st.session_state and st.session_state['materials_processed']:
    #outputs = generate_quiz(text)
    
    
    # render summary page
    # if page == 'Summary':
    #     summary = outputs[1]
    #     st.write(summary)

    # # render glossary page
    # elif page == 'Glossary':
    #     glossary = outputs[2]
    #     st.write(glossary)

    #render quiz page
        
        
        
    # if  st.session_state.quesai=='Yes':
    #     files = st.sidebar.file_uploader('Upload notes or lecture slides', accept_multiple_files=True, type=['pdf', 'docx', 'pptx', 'txt', 'md'])
    #     learning_files = files if files is not None else []
    #     st.session_state.text, documents = chat.load_documents(learning_files)
    #     print("Inside Question Generation Bot")
    #     docsearch = chat.get_cache_vectorstore(st.session_state.text,st.session_state.filename[0])
    #     print("embeddings Done")
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.session_state.complexity =  st.selectbox('Complexity Mode Required?*', ['No', 'Yes'],index=0,key="mode")
    #         st.session_state.no_of_questions = st.number_input('No. of  Questions to generate*',key="ai_questions",step=1)
    #         st.session_state.mode_of_questions = st.selectbox('Choose Answer Required/Not*', ['Only Questions', 'Questions with Answers'],index=0,key="quesansw")
            
            

    #     with col2:
    #         st.session_state.topic_name = st.text_input('Specific Chapter/Topic Name(Optional)',placeholder="AI Chapter/Topic Name")
    #         st.session_state.type_of_questions =  st.selectbox('Choose Question Type*', ['Short Questions', 'Long Questions','MCQ','Fill in the Blanks','True and False'],index=0)
    #         st.session_state.language =  st.selectbox('Choose Response Language Mode*', ['None','Punjabi','Hindi'],index=0,key="lang")
    #         #docsearch = chat.create_doc_embeddings(documents)
        
    #     # Storing the chat
    #     if 'generated' not in st.session_state:
    #         st.session_state['generated'] = []

    #     if 'past' not in st.session_state:
    #         st.session_state['past'] = []

    #     # Define a function to clear the input text
    #     def clear_input_text():
    #         global input_text
    #         input_text = ""

    #     # We will get the user's input by calling the get_text function
    #     def get_text():
    #         global input_text
    #         input_text = st.chat_input("Ask a question!")
    #         return input_text

    #     user_input = get_text()

    #     print('get the input text')

    #     # Initialize chat history
    #     if "messages" not in st.session_state:
    #         st.session_state.messages = []

    #     # Display chat messages from history on app rerun
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):
    #             if message['content'] != user_input:
    #                 st.markdown(message["content"])

    #     st.session_state.llm = load_chain(docsearch)
    #     print('chain loaded')
    #     memory = ConversationBufferMemory(
    #     return_messages=True,
    #     )
    #     st.session_state.language_chain = ConversationChain( llm=AzureChatOpenAI(
    #         deployment_name="gpt-4",
    #         temperature=0
    #         ),memory=memory)
    #     _ = st.session_state.llm({'question':initialise_prompt})
    #     if user_input:
    #         english_output,trans_output = chat.answer(user_input)
    #         print('get the output')
    #         st.session_state.messages.append({"role": "user", "content": user_input})
    #         # Display user message in chat message container
    #         with st.chat_message("user"):
    #             st.markdown(user_input)  

    #         # Display assistant response in chat message container
    #         with st.chat_message("assistant"):
    #             message_placeholder = st.empty()
    #             full_response = ""
    #             for char in english_output:
    #                 full_response += char
    #                 message_placeholder.markdown(full_response + "▌")
    #             message_placeholder.markdown(full_response)

    #             markdown_to_pdf(full_response,'question.pdf')
    #             st.download_button(
    #                 "Download",
    #                 data=open("question.pdf", "rb").read(),
    #                 file_name="question.pdf",
    #                 mime="application/pdf",
    #                 help="Download the questions as a PDF file"
    #             )


    #         st.session_state.messages.append({"role": "assistant", "content": english_output})

    #         # Display assistant response in chat message container

    #         if trans_output!="":
    #             with st.chat_message("assistant"):
    #                 message_placeholder = st.empty()
    #                 full_response = ""
    #                 for char in trans_output:
    #                     full_response += char
    #                     message_placeholder.markdown(full_response + "▌")
    #                 message_placeholder.markdown(full_response)

    #                 markdown_to_pdf(full_response,'question.pdf')
    #                 st.download_button(
    #                 "Download",
    #                 data=open("question.pdf", "rb").read(),
    #                 file_name="question.pdf",
    #                 mime="application/pdf",
    #                 help="Download the questions as a PDF file"
    #                 )

    #             st.session_state.messages.append({"role": "assistant", "content": full_response})

    # # if mode == 'AI Interactive':
    #     docsearch = chat.create_doc_embeddings(documents)

    #     # Storing the chat
    #     if 'generated' not in st.session_state:
    #         st.session_state['generated'] = []

    #     if 'past' not in st.session_state:
    #         st.session_state['past'] = []

    #     # Define a function to clear the input text
    #     def clear_input_text():
    #         global input_text
    #         input_text = ""

    #     # We will get the user's input by calling the get_text function
    #     def get_text():
    #         global input_text
    #         input_text = st.chat_input("Ask a question!")
    #         return input_text

    #     user_input = get_text()

    #     # Initialize chat history
    #     if "messages" not in st.session_state:
    #         st.session_state.messages = []

    #     # Display chat messages from history on app rerun
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):
    #             if message['content'] != user_input:
    #                 st.markdown(message["content"])

    #     if user_input:
    #         output = chat.answer(user_input, docsearch)
    #         st.session_state.messages.append({"role": "user", "content": user_input})
    #         # Display user message in chat message container
    #         with st.chat_message("user"):
    #             st.markdown(user_input)

    #         # Display assistant response in chat message container
    #         with st.chat_message("assistant"):
    #             message_placeholder = st.empty()
    #             full_response = ""
    #             for char in output:
    #                 full_response += char
    #                 message_placeholder.markdown(full_response + "▌")
    #             message_placeholder.markdown(full_response)

    #         st.session_state.messages.append({"role": "assistant", "content": full_response})

