import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.utils import read_file

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key = key, model_name = 'gpt-3.5-turbo', temperature = 0.6)

Template = """
Text: {text}
You are an expert MCQ generator and a subject-matter expert on {subject}, given the text, it is your responsibility to\
create a quiz of {number} multiple choice questions for {subject} students in a {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to also format your response like RESPONSE_JSON defined below and use it as a guide.\
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=['text','number','subject','tone','response_json'],
    template=Template
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt,output_key='quiz',verbose=True)

Template_2 = """
You are an expert english gramarian and writer. Given the Multiple Choice Quiz for {subject} students.
You need to evaluate the complexity of the question and give the complete analysis of the quiz. Only use at max 50 words for complexity
if the quiz is not on par with the cognitive and analytical abilities of the students,\
Update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student's ability
Quiz_MCQs:
{quiz}

Check from an expert English writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(input_variables=['subject','quiz'], template = Template_2)
quiz_review_chain = LLMChain(llm=llm, prompt = quiz_evaluation_prompt, output_key='review',verbose=True)

generate_evaluate_chain = SequentialChain(chains=[quiz_chain,quiz_review_chain],input_variables= ['text','number','subject','tone','response_json'],
                                          output_variables = ['quiz','review'], verbose = True)


#Function to extract data from text
def extracted_data(pages_data):

    template = """Extract all the following values : invoice no., Description, Quantity, date, 
        Unit price , Amount, Total, email, phone number and address from this data: {pages}

        Expected output: remove any dollar symbols {{'Invoice no.': '1001329','Description': 'Office Chair','Quantity': '2','Date': '5/4/2023','Unit price': '1100.00','Amount': '2200.00','Total': '2200.00','Email': 'Santoshvarma0988@gmail.com','Phone number': '9999999999','Address': 'Mumbai, India'}}
        """
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)

    llm = OpenAI(temperature=.7)
    full_response=llm(prompt_template.format(pages=pages_data))
    

    #The below code will be used when we want to use LLAMA 2 model,  we will use Replicate for hosting our model...
    
    #output = replicate.run('replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1', 
                           #input={"prompt":prompt_template.format(pages=pages_data) ,
                                  #"temperature":0.1, "top_p":0.9, "max_length":512, "repetition_penalty":1})
    
    #full_response = ''
    #for item in output:
        #full_response += item
    

    #print(full_response)
    return full_response


def create_docs(user_pdf_list):
    
    df = pd.DataFrame({'Invoice no.': pd.Series(dtype='str'),
                   'Description': pd.Series(dtype='str'),
                   'Quantity': pd.Series(dtype='str'),
                   'Date': pd.Series(dtype='str'),
	                'Unit price': pd.Series(dtype='str'),
                   'Amount': pd.Series(dtype='int'),
                   'Total': pd.Series(dtype='str'),
                   'Email': pd.Series(dtype='str'),
	                'Phone number': pd.Series(dtype='str'),
                   'Address': pd.Series(dtype='str')
                    })

    for filename in user_pdf_list:
        
        print(filename)
        raw_data=get_pdf_text(filename)
        #print(raw_data)
        #print("extracted raw data")

        llm_extracted_data=extracted_data(raw_data)
        #print("llm extracted data")
        #Adding items to our list - Adding data & its metadata

        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            # Converting the extracted text to a dictionary
            data_dict = eval('{' + extracted_text + '}')
            print(data_dict)
        else:
            print("No match found.")

        
        df=df.append([data_dict], ignore_index=True)
        print("********************DONE***************")
        #df=df.append(save_to_dataframe(llm_extracted_data), ignore_index=True)

    df.head()
    return df