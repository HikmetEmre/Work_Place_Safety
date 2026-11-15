"""Streamlit app: classify workplace accident reports as Fatal or Nonfatal.

Run with:  streamlit run app_workpl.py
"""
from datetime import date, datetime

import pandas as pd
import streamlit as st
from joblib import load

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
REPO_RAW = "https://raw.githubusercontent.com/HikmetEmre/Work_Place_Safety/main"
SAMPLE_DATA_PATH = "for_app.csv"
VECTORIZER_PATH = "cv1_model.pkl"
CLASSIFIER_PATH = "mnb11_model.pkl"

FATAL_IMAGE = (
    "https://us.123rf.com/450wm/elartico/elartico1703/elartico170300026/"
    "73175348-danger-sign-with-skull-symbol.-deadly-danger-sign,-warning-sign..jpg"
)
NONFATAL_IMAGE = (
    "https://us.123rf.com/450wm/get4net/get4net2111/get4net211121281/"
    "177820679-warning-signal-for-road-hazard-and-public-safety.jpg?ver=6"
)


# --------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load the fitted CountVectorizer and MultinomialNB classifier once."""
    return load(VECTORIZER_PATH), load(CLASSIFIER_PATH)


@st.cache_data
def load_sample_data():
    """Load the sample table shown on the page."""
    return pd.read_csv(SAMPLE_DATA_PATH)


def preprocess_text(text):
    """Normalise a report before vectorizing (lowercase, trim)."""
    return text.lower().strip()


def predict_level(text):
    """Return the predicted damage level ('Fatal' / 'Nonfatal') for a report."""
    vectorizer, classifier = load_models()
    features = vectorizer.transform([preprocess_text(text)])
    return classifier.predict(features)[0]


# --------------------------------------------------------------------------
# Page sections
# --------------------------------------------------------------------------
def render_page_config():
    st.set_page_config(
        page_title="IncidentFatalityPredictor",
        page_icon="https://i.pinimg.com/originals/fb/d1/e9/fbd1e95fc924fd69a44cf8fb27c47683.jpg",
        menu_items={
            "Get help": "mailto:hikmetemreguler@gmail.com",
            "About": "For More Information\n"
            "https://github.com/HikmetEmre/Work_Place_Safety",
        },
    )


def render_intro():
    st.title("**:red[NLP and EDA Work on Workplace Accident Reports.]**")
    st.markdown(
        "**Introducing :red[AccidentSense] A powerful NLP-based model for classifying "
        "workplace accident reports as fatal or nonfatal, enabling effective risk "
        "assessment and proactive safety measures.**."
    )
    st.image(f"{REPO_RAW}/model_robot.jpg")

    st.markdown(
        "**This project involved the analysis of OSHA Accident and Injury Data, which "
        "consists of reports and various features related to accidents and injuries.**"
    )
    st.markdown(
        "**The dataset, containing 35k records, was extensively analyzed to extract "
        "valuable insights such as the top ten injury sources, the most affected body "
        "parts, the distribution of accidents across days of the week, months, and "
        "hours of the day, among other visualizations.**"
    )
    st.markdown(
        "**Furthermore, NLP techniques were applied to analyze the word frequency in "
        "accident reports, and polarity (fatal or nonfatal) was assigned based on the "
        "content. A machine learning model was trained and tested to classify new "
        "accident reports accurately.**"
    )
    st.markdown(
        "The project showcases a comprehensive approach combining data analysis, NLP, "
        "and predictive modeling to enhance safety measures and risk assessment in "
        "workplace environments."
    )
    st.markdown(
        "*:red[Alright, Put on your gloves and safety glasses; we are going into the field.]*"
    )
    st.image(f"{REPO_RAW}/examine2.jpg")


def render_metadata():
    st.header("**META DATA**")
    st.markdown("- **Text**:A written or textual expression that provides a detailed description of accident.")
    st.markdown("- **Level**: The Damage Level of Workplace Accident ")
    st.markdown("- **Polarity**:Typically ranging from -1 to +1, where negative values indicate negative sentiment, positive values indicate positive sentiment.")
    st.markdown("- **Sentiment**: Sentiment refers to the emotional or subjective attitude expressed in text.")

    st.table(load_sample_data().sample(5, random_state=17))
    st.image(f"{REPO_RAW}/accident_event_word_cloud.png")


def render_sidebar():
    """Draw the sidebar and return (report_text, submit_clicked)."""
    st.sidebar.markdown(
        "**INPUT** The **:blue[Text]** Of Report to see the Result Of **:red[Damage Level]**"
    )
    report_text = st.sidebar.text_input(
        "**:red[A text as Abstract Text of Accident Report.]**"
    )
    return report_text, st.sidebar.button("Submit")


def render_results(report_text, submitted):
    st.header("Results")

    if not submitted:
        st.markdown("Please click the *Submit Button*!")
        return
    if not report_text.strip():
        st.warning("Please enter the text of an accident report first.")
        return

    level = predict_level(report_text)

    st.info("You can find the result below.")
    st.table(
        pd.DataFrame(
            {
                "Date": [date.today()],
                "Time": [datetime.now().strftime("%H:%M:%S")],
                "Text": [report_text],
                "Level Of Accident Damage": [level],
            }
        )
    )

    if level == "Fatal":
        st.image(FATAL_IMAGE)
    elif level == "Nonfatal":
        st.image(NONFATAL_IMAGE)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    render_page_config()
    render_intro()
    render_metadata()
    report_text, submitted = render_sidebar()
    render_results(report_text, submitted)


main()
