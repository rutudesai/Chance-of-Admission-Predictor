#Importing Libraries:
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
#import time

#This part of code creates a sidebar and even gives it a heading 
st.sidebar.header('User Input Parameters')

Page=st.sidebar.selectbox('Select Page:',['Home Page','About the Data','Visualization','Predictor'])

if(Page=='Home Page'):
    #The below code is used to write stuff on Web App
    st.write("""
    # Predictiction of Admittance for Masters Aspirants

    This app predicts the Chances of being admitted in **_Ivy League_** Colleges
    """)
    #The below code will show an image and below it there will be caption
    image=Image.open('Colleges.JPG')
    st.image(image)



if(Page=='About the Data'):
    st.write("""
    # Chance of Admit-Predictor

    This app predicts the Chances of being admitted in **_Ivy League_** Colleges
    """)
    st.write("""
    # About The Data
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    st.write("""

    **Context**

    This dataset is created for prediction of Graduate Admissions from an **_Indian perspective_**.

    **Content**

    The dataset contains several parameters which are considered important during the application for Masters Programs.
    The parameters included are :

    GRE Scores ( out of 340 )

    TOEFL Scores ( out of 120 )

    University Rating ( out of 5 )

    Statement of Purpose and Letter of Recommendation Strength ( out of 5 )

    Undergraduate GPA ( out of 10 )

    Research Experience ( either 0 or 1 )

    Chance of Admit ( ranging from 0 to 1 )

    **Acknowledgements**

    This dataset is inspired by the **_UCLA Graduate Dataset_**. The test scores and GPA are in the older format.
    The dataset is owned by **_Mohan S Acharya_**.

    **Inspiration**

    This dataset was built with the purpose of helping students in shortlisting universities with their profiles. The predicted output gives them a fair idea about their chances for a particular university.
    
    **Citation**

    Please cite the following if you are interested in using the dataset :
    Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019

    """)

if(Page=='Predictor'):
    st.write("""
    # Chance of Admit-Predictor

    This app predicts the Chances of being admitted in **_Ivy League_** Colleges
    """)
    st.write("""
    # Predictor
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    #Creting sliders for every feature to be used
    def user_input_features():
        GRE_Score=st.sidebar.slider('Gre Score',260,340)
        TOEFL_Score=st.sidebar.slider('TOEFL Score',90,120)
        CGPA=st.sidebar.slider('CGPA',6.0,10.0)
        #University_Rating=st.sidebar.slider('University Rating',1.0,5.0)
        #LOR=st.sidebar.slider('LOR',1.0,5.0)
        #SOP=st.sidebar.slider('SOP',1.0,5.0)
        #Research=st.sidebar.slider('Research Done',0,1)
        data={'GRE Score':GRE_Score,
          'TOEFL Score':TOEFL_Score,
          'CGPA':CGPA
          #'University Rating':University_Rating,
          #'LOR':LOR,
          #'SOP':SOP,
          #'Research':Research
          }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()
    
    #Loading the data and cleaning it
    data_path=r'Admission_Predict_Ver1.csv'
    data=pd.read_csv(data_path)
    data.drop(['Serial No.'],axis=1,inplace=True)
    data.rename(columns={'LOR ':'LOR','Chance of Admit ':'Chance of Admit'},inplace=True)


    #Preparing the data for Model
    X=data[['GRE Score','TOEFL Score','CGPA']]
    y=data.iloc[:,-1]


    #from sklearn.model_selection import train_test_split
    #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    #Applying Linear Regression Model
    model1=LinearRegression()
    #model.fit(X_train,y_train)
    #y_pred=model.predict(X_test)
    #acc=r2_score(y_test,y_pred)
    #st.write('The Accuracy is :',acc)
    #st.write(model.coef_)
    #st.write(model.intercept_)
    model1.fit(X,y)
    st.subheader('User Input parameters')
    st.write(df)
    y_val=model1.predict(df)
    if(y_val*100 <100):
        #with st.spinner('Wait for it...'):
        #time.sleep(1)
        #st.success('Done!')
        st.markdown('The Chances of Admission are:**_{}%_**'.format(y_val*100))
    else:
        st.markdown('The Chances of Admission are:**_100%_**')

            
    #model2=RandomForestRegressor(n_estimators=100,random_state=0)
    #model2.fit(X,y)
    #st.subheader('User Input parameters')
    #st.write(df)
    #y_val=model2.predict(df)
    #st.write('The Chance of Admission is:',y_val)


    if st.button('Conclusion'):
        st.write("""
        # Conclusion
        """)
        if(y_val*100>=80):
            st.write('Congratulations!! You have great chances of getting Admitted!  :)')
        else:
            st.write('You have to do more Hardwork and get better scores and CGPA next time!  :(')

if(Page=="Visualization"):
    st.write("""
    # Chance of Admit-Predictor

    This app predicts the Chances of being admitted in **_Ivy League_** Colleges
    """)
    st.write("""
    # Visualization

    
    """)
    #The below code will show an image and below it there will be caption
    
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    image=Image.open('1.JPG')
    st.image(image,caption='Distribution of Chance of Admit using KDE Distplot and Boxplot')
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    image=Image.open('heatmap.png')
    st.image(image,caption='Correlation Between differnt Features')
    st.write("""
    According to the Heatmap ,it is clear that CGPA ,GRE Score and TOEFL Score are most important features related to being Admitted
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")


    image=Image.open('2.JPG')
    st.image(image,caption='How are differnt Features distributed')
    st.write("""
    Distplots showing Distribution of the main Features.
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")


    image=Image.open('3.JPG')
    st.image(image,caption='Number of Students who have done Research or not')
    st.write("""
    We can see that majority of the students have written Research Papers!
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    
    image=Image.open('4.JPG')
    st.image(image,caption='Correlation Between CGPA,TOEFL Score and GRE Score')
    st.write("""
    The Plot is generally showing that students having good CGPA have good GRE and TOEFL Scores and vice versa.
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    
    image=Image.open('5.JPG')
    st.image(image,caption='Do Students with Good CGPA write Research Paper?')
    st.write("""
    Yes, the students who have more CGPA have written Research Papers
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")
    
    image=Image.open('6.JPG')
    st.image(image,caption='Correlation Between Chance of Admit,University Rating,LOR and SOP')
    st.write("""
    The Strip Plot is clearly showing that students studying in Good Universities have better LOR,SOP and Chances of Admit are definitely high!
    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    st.write("""

    The **3** Parameters which are the most important in getting Admission are:

    1)**GRE Score**

    2)**TOEFL Score**

    3)**CGPA**

    """)
    st.write("""------------------------------------------------------------------------------------------------------------------------------------""")

    

st.write('**Made with Streamlit**-By:**Rutu Desai**')

