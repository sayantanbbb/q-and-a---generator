
import streamlit as st
from simplet5 import SimpleT5
import random
import string
import numpy as np
from transformers import pipeline



def specific_string(length):  
     
    letters = string.ascii_lowercase # define the lower case string  
     # define the condition for random.choice() method  
    result = ''.join((random.choice(letters)) for x in range(length))  
    print(" Random generated string with repetition: ", result)  


def extract_main_thing(para,sep=" "):
    return " ".join(para.split(sep)[1:])
@st.cache(hash_funcs={"simplet5.simplet5.SimpleT5":id,"transformers.pipelines.question_answering.QuestionAnsweringPipeline":id})
def load_models():
    model2 = SimpleT5()
     

    model2.load_model("t5","../input/training-t5-model-yet-again/outputs/simplet5-epoch-0-train-loss-1.9395-val-loss-1.9168",use_gpu=True)
    
    nlp = pipeline("question-answering")
    return model2,nlp
model2,nlp=load_models()
@st.cache()
def saving_temporary_questions():
    return {"questions":[],
            "answers":[],
           "prev":"",
           "slider":0,
           "stride-amount":70}
qa=saving_temporary_questions()
class SlidingWindow():
    def __init__(self,words,stride,window):
        self.idx1=0 
        self.idx2=window 
        self.stride=stride
        self.words=words
    def get_item(self):
        if self.idx2-len(self.words)>=self.stride:
            return "END"
        output=self.words[self.idx1:min([len(self.words),self.idx2])] 
        self.idx1+=self.stride 
        self.idx2+=self.stride  
        return output 
def create_questions(text,number,stride_percent=70):
    words=text.split()  
    window_size=int(len(words)/number)
    stride=max([2,int(window_size*stride_percent/100)])
    slided=SlidingWindow(words,stride,window_size) 
    questions=[]
    answers=[]
    while True: 
        item=slided.get_item()
        if item=="END":
            break 
        questions+=model2.predict(" ".join(item))
        answers.append(nlp(question=questions[-1],context=text)['answer'])
    return questions,answers
        
        
st.title("AI powered question answer generator")
input_text=st.text_area("Input your passage :","your passage to generate questions from here") 
no = st.slider('number of questions you want too many might harm performance', 0, 0,50)
stri=st.slider("stride percentage",0,0,100)
if st.checkbox("generate questions"): 
    st.text("working on it")
    question_elements=[] 
    answer_elements=[] 
    if qa["prev"]!=input_text or qa["slider"]!=int(no) or qa["stride-amount"]!=int(stri):
        qa["questions"],qa["answers"]=create_questions(input_text,no,int(stri))
        qa["prev"]=str(input_text)
        qa["slider"]=int(no)
        qa["stride-amount"]=int(stri)
        
    column1,column2=st.columns(2)
    mask=list([1 for _ in qa["questions"]]) 
    
    with column1:
            st.text("\n \n")
            i=0
            for question,answer in zip(qa["questions"],qa["answers"]):
                quee=st.text_input("question:-",value=f"{i}) {question}",key=str(i)+"q")
                anse=st.text_input("answer:-",value=f"answer {answer}",key=str(i)+"a")
                i+=1
                question_elements.append(quee)
                answer_elements.append(anse)
    with column2:
            checks=[]
            st.text("Check a question to delete it")
            for i,_ in enumerate(qa["questions"]):
                checks.append(st.checkbox(f"~ {i}",key=str(i)))
                for __ in range(10):st.text("")
    for index,element in enumerate(question_elements):
        print(str(element))
        if f"{qa['questions'][index]}" not in str(element):
            qa["questions"][index]=extract_main_thing(str(element))
     
    for index,element in enumerate(answer_elements):
        if str(element)!=f"answer {qa['answers'][index]}":
            qa["answers"][index]=extract_main_thing(str(element),"answer")
         
    for index,c in enumerate(checks):
        if c:
            mask[index]=0 
       
    d1=st.checkbox("Download with answers")
    d2=st.checkbox("Download only questions") 
    lines=[]
    if d1: 
        answers_m,questions_m=[],[]
        for i in range(len(mask)):
            if mask[i]==1: 
                answers_m.append(qa["answers"][i])
                questions_m.append(qa["questions"][i])
            
        for question,answer in zip(questions_m,answers_m):
            lines.append("question :- {}".format(question))
            lines.append("answer :- {}".format(answer))
        document=("\n".join(lines)).encode("utf-8")
        st.download_button("click here to download your file [answers included]!",data=document,file_name="q and a.txt")
    if d2: 
        questions_m=[]
        for i in range(len(mask)):
            if mask[i]==1: 
          
                questions_m.append(qa["questions"][i])
            
        for question in questions_m:
            lines.append("question :- {}".format(question))
       
        document=("\n".join(lines)).encode("utf-8")
        st.download_button("click here to download your file [answers excluded]!",data=document,file_name="q and a.txt")
        
