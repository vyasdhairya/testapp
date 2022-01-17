import streamlit as st
import pandas as pd
import hashlib
from PIL import Image
from preprocess1 import prepro1
from preprocess2 import prepro2
#import numpy as np
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(FirstName TEXT,LastName TEXT,Mobile TEXT,Email TEXT,password TEXT,Cpassword TEXT)')
def add_userdata(FirstName,LastName,Mobile,Email,password,Cpassword):
    c.execute('INSERT INTO userstable(FirstName,LastName,Mobile,Email,password,Cpassword) VALUES (?,?,?,?,?,?)',(FirstName,LastName,Mobile,Email,password,Cpassword))
    conn.commit()
def login_user(Email,password):
    c.execute('SELECT * FROM userstable WHERE Email =? AND password = ?',(Email,password))
    data = c.fetchall()
    return data
def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def main():
    st.title("Welcome To Ticket AIssistant")
    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        original_title="<p style='text-align: center;'>In this corporate world to deal with tedious work of higher numbers of tickets and bugs it is difficult task web and saas companies to manually work on it. So to overcome it we have developed web based application which automatically separates ticket and bug with the help of different machine learning techniques.</p>"
        image = Image.open('flow.jpg')
        st.image(image)
        #st.image(np.array([cv2.imread("flow.jpg")]), channels="BGR")
        st.markdown(original_title, unsafe_allow_html=True)
    elif choice == "Login":
        st.subheader("Login Section")
        Email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(Email,check_hashes(password,hashed_pswd))
            if result:
                st.success("Logged In as {}".format(Email))
                task = st.selectbox("Selection Option",["Text Query","Upload .CSV"])
                if task == "Text Query":                    
                    ab1=st.text_input("Write Text in English")
                    clss=prepro1(ab1)
                    listToStr = ' '.join([str(elem) for elem in clss])
                    task2 = st.selectbox("Selection ML",["Support Vector Machine","K-Nearest Neighbor","Naive Bayes","Random Forest","Decision Tree","Extra Tree"])
                    if st.button("Classify"):                        
                        st.success('The Query is '+listToStr)
                elif task == "Upload .CSV":
                    st.subheader("Upload .CSV File Only")
                    uploaded_file = st.file_uploader("Choose a file")
                    dataframe = pd.read_csv(uploaded_file)
                    st.dataframe(dataframe, 500, 500)
                    clss,dff=prepro2(dataframe)
                    listToStr = ','.join([str(elem) for elem in clss])
                    task2 = st.selectbox("Selection ML",["Support Vector Machine","K-Nearest Neighbor","Naive Bayes","Random Forest","Decision Tree","Extra Tree"])
                    if st.button("Classify"):                        
                        st.dataframe(dff, 500, 800)
                        
            else:
                st.warning("Incorrect Email/Password")
                
    elif choice == "SignUp":
        FirstName = st.text_input("Firstname")
        LastName = st.text_input("Lastname")
        Mobile = st.text_input("Mobile")
        Email = st.text_input("Email")
        new_password = st.text_input("Password",type='password')
        Cpassword = st.text_input("Confirm Password",type='password')
        if st.button("Signup"):
            create_usertable()
            add_userdata(FirstName,LastName,Mobile,Email,make_hashes(new_password),make_hashes(Cpassword))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
            
if __name__ == '__main__':
    main()