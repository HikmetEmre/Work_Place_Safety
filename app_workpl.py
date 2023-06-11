### New App For Work Accidents Classify ###
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

#### Page Config ###
st.set_page_config(
    page_title="IncidentFatalityPredictor",
    page_icon="https://i.pinimg.com/originals/fb/d1/e9/fbd1e95fc924fd69a44cf8fb27c47683.jpg",
    menu_items={
        "Get help": "mailto:hikmetemreguler@gmail.com",
        "About": "For More Information\n" + "https://github.com/HikmetEmre/Work_Place_Safety"
    }
)

### Title of Project ###
st.title("**:red[NLP and EDA Work on Workplace Accident Reports.]**")

### Markdown ###
st.markdown("**Introducing :red[AccidentSense] A powerful NLP-based model for classifying workplace accident reports as fatal or nonfatal, enabling effective risk assessment and proactive safety measures.**.")

### Adding Image ###
st.image("https://raw.githubusercontent.com/HikmetEmre/Work_Place_Safety/main/model_robot.jpg")

st.markdown("**This project involved the analysis of OSHA Accident and Injury Data, which consists of reports and various features related to accidents and injuries.**")
st.markdown("**The dataset, containing 35k records, was extensively analyzed to extract valuable insights such as the top ten injury sources, the most affected body parts, the distribution of accidents across days of the week, months, and hours of the day, among other visualizations.**")
st.markdown("**Furthermore, NLP techniques were applied to analyze the word frequency in accident reports, and polarity (fatal or nonfatal) was assigned based on the content. A machine learning model was trained and tested to classify new accident reports accurately.**")
st.markdown("The project showcases a comprehensive approach combining data analysis, NLP, and predictive modeling to enhance safety measures and risk assessment in workplace environments.")

st.markdown("*:red[Alright, Put on your gloves and safety glasses; we are going into the field.]*")

st.image("https://raw.githubusercontent.com/HikmetEmre/Work_Place_Safety/main/examine2.jpg")

#### Header and definition of columns ###
st.header("**META DATA**")



st.markdown("- **Text**:A written or textual expression that provides a detailed description of accident.")
st.markdown("- **Level**: The Damage Level of Workplace Accident ")
st.markdown("- **Polarity**:Typically ranging from -1 to +1, where negative values indicate negative sentiment, positive values indicate positive sentiment.")
st.markdown("- **Sentiment**: Sentiment refers to the emotional or subjective attitude expressed in text.")


### Example DF ON STREAMLIT PAGE ###
df=pd.read_csv('for_app.csv')


### Example TABLE ###
st.table(df.sample(5, random_state=17))

st.image("https://raw.githubusercontent.com/HikmetEmre/Work_Place_Safety/main/accident_event_word_cloud.png")

#---------------------------------------------------------------------------------------------------------------------

### Sidebar Markdown ###
st.sidebar.markdown("**INPUT** The **:blue[Text]** Of Report to see the Result Of **:red[Damage Level]**")

### Define Sidebar Input's ###
ReportText = st.sidebar.text_input("**:red[A text as Abstract Text of Accident Report.]**")


#---------------------------------------------------------------------------------------------------------------------

from joblib import load

nlp_model = load('mnb11_model.pkl')
cv1 = load('cv1_model.pkl')

# Function to preprocess the input data
def preprocess_data(text):
    # Apply any necessary preprocessing steps here
    preprocessed_text = text.lower()  # Example: Convert text to lowercase
    return preprocessed_text

preprocessed_report = preprocess_data(ReportText)

transformed_report = cv1.transform([preprocessed_report])




    

pred = nlp_model.predict(transformed_report)





#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

### Result Screen ###
if st.sidebar.button("Submit"):

    ### Info message ###
    st.info("You can find the result below.")

    ### Inquiry Time Info ###
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    ### For showing results create a df ###
    results_df = pd.DataFrame({
    'Date': [today],
    'Time': [time],
    'Text': [ReportText],
    'Level Of Accident Damage': [pred]
    })

   


    st.table(pred)

    if pred == 'Fatal':
        st.image("https://us.123rf.com/450wm/elartico/elartico1703/elartico170300026/73175348-danger-sign-with-skull-symbol.-deadly-danger-sign,-warning-sign..jpg")

    elif pred == 'Nonfatal':
        st.image("https://us.123rf.com/450wm/get4net/get4net2111/get4net211121281/177820679-warning-signal-for-road-hazard-and-public-safety.jpg?ver=6")
      
else:
    st.markdown("Please click the *Submit Button*!")
