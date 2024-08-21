import logging

logger = logging.getLogger(__name__)


'''SYSTEM_PROMPTS = {
                   "Default": {
                   "role": "system",
                   "content": 
                    """
                    You are a New Generation AI based School Teacher who is expert in school subjects. 
                    Your job is to generate questions and answers basis on the query of user. 
                    Generate specific numbers of questions as asked by the User. 
                    If complexity is not specified, generate a mix of easy, moderate and hard questions for the asked Topic. 
                    Format the questions in a clear and organized manner. 
                    Generate the questions always and generate answers only when asked in User's query.
                    
                    """
                    },
                }
'''

initialise_prompt =""" You are an educator for class 1 to class 12 for Madhya Pradesh School Education Board, 
                       India. Answer in a clear and organized manner.
                       Generate the Question and Answer from the uploaded document only, 
                       Don't take help from the internet.
                    """



filter_prompt2 = """ For the following user query, 
                     first analyze whether it is related to {} books/chapters.
                     If the query is related then Based on the context answer the user query else return message "This is beyond my capabilites, please ask another question."
                     User Query - {}
                 """

# format_prompt=""" Format the response in proper question format and shuffule the questions.Remember to keep all the information as it is. Response - {}"""

# format_prompt=""" Format the response and provide only questions in both the language which are there in response and shuffle the response and consider following constraints,
# 1. number of questions - {}
# 2. type of questions - {}
# 3. Topic name - {} 

# provide exact number of questions as per provided
# convert the mathematical expressions in Latex form.
# Remember to keep all the information as it is. Response - {}"""

# format_prompt="""Response which is given is in dictionary format . Please convert these into proper question paper format 
# Restrict the response to {} number of questions and reorder the questions.

# Shuffle the order of questions

# Keep the mathematical expressions in LaTex format 
# Remember to keep all the information as it is. Response - {}
# """

# format_prompt="""From the context Provide the summary in tabular format of Response {} contain following parameters
#     1. Topics name
#     2. Total Number of Questions
#     3. Type of Questions

#     Provide the Topic names and Type of Questions in proper tabular format 
#     Also mention the total number of questions and class at the end 
# """

format_prompt="""format and regenerate the Response {}"""


aiformat_prompt="""From the context Provide the brief summary in tabular format of Response {} contain following 
                    1. Chapter names
                    2. Description of each topic

                    provide the response in tabular format . 
                    Only include Chapter names and Description
                    Also mention the subject at the end 
                    not repeated multiples time just metioned 4 to 5 rows only .
                """

latex_prompt="""format the response {} into python list.
                Remember to keep all the information as it is
             """

#ai_prompt="""Based on the context, generate only questions considering following constraints,
#             1. number of questions  - {}
#             2. mode of questions - {}
#             3. type of questions - {} 
#             Generate questions according to Madhya Pradesh Board Education School
#             Response is to be generated in both english and Hindi,first generate in english then in Hindi
#             and if you are generating Answers also, Answer start in a new line and Generate questions according to Madhya Pradesh School Education Board
#             generate the all types of question and answer,and please insure the question and answer based on topic or text.
#             Remember to keep all the information as it is. Response - {}
#"""

ai_prompt="""Based on the context, generate only questions considering following constraints,
             1. number of questions  - {}
             2. mode of questions - {}
             3. type of questions - {} 
             Generate questions in both English and Hindi,first generate in English then in Hindi. The generated questions must be numbered seperately for English and Hindi.
             If you are generating Answers also, answer start from a new line.
             Generate all types of question and answer,and please insure the question and answer based on topic or text.
             Remember to keep all the information as it is. Response - {}
"""

ai_topic_prompt="""Based on the context, generate only questions considering following constraints,
                    1. topic of questions  - {}
                    2. number of questions - {} 
                    3. mode of questions - {}
                    4. type of questions - {}
                    5. Level of questions - {}
                    Response is to be generated in both english and Hindi,first generate in english then in Hindi,The generated questions must be numbered seperately for English and Hindi.
                    and if you are generating Answers also, Answer it in a new line and Generate questions according to Madhya Pradesh School Education Board,
                    generate the all types of question and answer,and please insure the question and answer based on topic or text.
                """


ai_topic_prompt1 = """
                    You are provided with a document containing text related to a specific topic. 
                    Response is to be generated in both English and Hindi, first generate in English then in Hindi. 
                    Ensure that all questions and answers, if required, are derived solely from the provided text. 
                    Follow these guidelines strictly:

                    
                    1. **Number of Questions**: {} (generate exactly this number of questions in the specified language(s))
                    2. **Text**: {}   
                    3. **Language**: {} (generate questions in the specified language(s))
                    4. **Mode of Questions** (Questions only / Questions with Answers): {}
                    5. **Type of Questions** (Short Questions / Long Questions / MCQ / Fill in the Blanks / True and False): {}
                    6. **Complexity Level** (Easy / Difficult): {}

                    ### Instructions:
                    - **Strictly use the content from the provided text** to formulate the questions and answers.
                    - **Do not use any external sources**.
                    - Generate **exactly** the specified number of questions in the chosen language(s).
                    - If answers are required, provide the answers immediately following each question.
                    - Format the response in the chosen language(s):
                    - If "English" is chosen, generate questions in English only.
                    - If "English and Hindi" is chosen, generate questions in both languages English and Hindi, first in English then in Hindi.
                    - Ensure clarity and readability in both questions and answers.
                    - For multiple-choice questions, provide 4 options, followed by the correct answer.

                    ### Example Format:
                    **Question 1:** [Your question here]
                    **a.** [Option 1]
                    **b.** [Option 2]
                    **c.** [Option 3]
                    **d.** [Option 4]
                    **Answer 1:** [Correct answer here]

                    **Question 1:** [Your question here]
                    **a.** [Option 1]
                    **b.** [Option 2]
                    **c.** [Option 3]
                    **d.** [Option 4]
                    **Answer 1:** [Correct answer here]

                    **Fill in the Blanks Example:**

                    **Question 1:** The ______ is black.
                    **Answer 1:** The cat is black.

                    **Question 1:** ______ काली है।
                    **Answer 1:** बिल्ली काली है।

                    **True/False Example:**
                    **Question 1 (English):**
                    [Your statement here]

                    **Answer 1 (English):**
                    [True/False] (if required)

                    **Question 1 (Hindi):**
                    [Your statement here]

                    **Answer 1 (Hindi):**
                    [True/False] (if required)

                    Generate exactly {} questions in the specified language(s) based on the above instructions.
                  """

ai_topic_prompt_questions= """ Based on the context, extract only questions
                               Remember to keep all the information as it is. Response - {}
                           """

mcq_test_prompt =          """ Based on the context generate terminologies and key terms
                               first generate terminologies then generate key terms
                               generate the response in both english and Hindi,first generate in english then in Hindi
                               Remember to keep all the information as it is. Response-{}    
                           """

key_term_prompt = """
                    Based on the context generate key terms
                    Remember to keep all the information as it is. Response-{}    
                  """

learn_outcome_prompt = """
                        Based on the context generate learning outcomes
                        generate the response in both english and Hindi, first generate in english then in Hindi
                        Remember to keep all the information as it is. Response-{} 
                            
                    """
student_prompt = """Based on query {} generate response from context {}"""


''' student_prompt = """You are an intelligent assistant designed to help students understand and extract information from their study materials. A student has uploaded a book/document and has asked a specific question about its content. Your task is to provide a clear and concise answer based on the provided text. Use the text from the document to formulate your response.
                    Instructions:
                    Document: []
                    Question: []
                    Provide an answer that directly addresses the student's question using the content from the document. 
                    If the question requires specific information or context, include relevant excerpts from the document to support your answer.

                    Example-

                    Document: [The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity. 
                    It consists of two main parts: special relativity and general relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light is constant regardless of the observer's motion. General relativity, published in 1915, expanded on this by describing gravity not as a force, but as a curvature of spacetime caused by mass and energy.]
                    
                    Question: What is the main difference between special relativity and general relativity?

                    Answer: The main difference between special relativity and general relativity lies in their treatment of gravity and reference frames. Special relativity, published in 1905, deals with the laws of physics being the same for all non-accelerating observers and asserts that the speed of light is constant regardless of motion. General relativity, published in 1915, goes further by describing gravity not as a force but as a curvature of spacetime caused by mass and energy.

                    If the question is not from the related document then say "Ask me something else."
                    if you don't know answer then help the internet.
            
              """
''' 

master_prompt = """
                Level of questions : This is the depth of the questions the teacher wants to generate. A low depth will cover the basics, and generalizations while a high depth will cover the specifics, details, unfamiliar, complex, and side cases. The lowest depth level is Easy, and the highest is Complex.
                Do not discuss about the context you have recieved and do not mention you are unable to access or read any documents 
                Easy : Covers basics questions with simple definitions and brief explanations, suitable for beginners or quick overviews.
                Difficult : Provides expert and complex questions .
                if type of questions is Short Questions then generate questions according to 1 marks and 2 marks from given context
                if type of questions is Long Questions then generate questions according to 3,4,5 marks from given context
                You have been provided the Context/Subject/Chapter/Book 
                Based on the context, generate {} considering following constraints,
                    1. Level of questions - {}
                    2.
                    3. number of questions  - {}
                    4. type of questions - {}
                after all the selected these thing than generate the question and answer according to selected document file only.
                If topic name is not provided then based on context generate {} from books/chapters/Subject 

                if Level of questions is Yes then categorise the Response as Easy or Difficult 

                if topic name is not from context then mention this topic name is out of context

                Remember to generate questions according to Madhya Pradesh School Education Board
                Remember to keep all the information as it is. Response - {}
                Remember to not discuss about the context you have recieved and do not mention you are unable to access or read any documents 

                """

lang_prompt = """ Translate the below response to {} language . 
                  Remember to keep all the information as it is. Response - {}
              """
