import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# Constants
MAX_GEN_LEN = 100
MAX_TUPLE_LEN = 4
MAX_CHARS = 200
COMMAND_PHRASE = 'cmd'
TEXT_FILE = 'all-inputs.txt'
ALLOWED_WORDS = set('''
a the man child wife husband car movie saw played ran sat with after from on
ball house park tree book message game rain jump eat drink sleep swim fly sing dance
chair table apple banana orange juice milk water bread cheese sky grass flower sun
moon star cloud snow rain light dark small big happy sad angry'''.split())
START_TOKEN = ' '
END_TOKEN = '.'

# Load training data
sentences = []
if os.path.isfile(TEXT_FILE):
    with open(TEXT_FILE, 'r') as file:
        sentences = [line.strip() for line in file.readlines()]

# Helper functions
def validate_sentence(sentence):
    words = sentence.replace(',', '').replace('.', '').replace('!', '').lower().split()
    invalid_words = [word for word in words if word not in ALLOWED_WORDS]
    return False if invalid_words else True

def add_training_data(sentence):
    if len(sentence) > MAX_CHARS:
        st.error(f'Your sentence exceeded {MAX_CHARS} characters.')
        return
    if validate_sentence(sentence):
        with open(TEXT_FILE, 'a') as file:
            file.write(' '.join(sentence.lower().split()) + '\n')
        st.success('Sentence added to training data.')
    else:
        st.error('Your sentence contains words not in the allowed list.')

def get_tuples(num_prev_words=1):
    tuples = []
    for sentence in sentences:
        words = [START_TOKEN] * num_prev_words + sentence.split() + [END_TOKEN]
        tuples.extend([words[i-num_prev_words:i+1] for i in range(num_prev_words, len(words))])
    columns = ['Next'] + [f'{i+1} words ago' for i in range(num_prev_words)]
    return pd.DataFrame(tuples, columns=columns[::-1])

def plot_frequencies(df):
    fig, ax = plt.subplots()
    df.plot(kind='bar', ax=ax)
    st.pyplot(fig)

def generate_sentence(df):
    n = len(df.columns) - 1
    sentence = [START_TOKEN] * n
    while sentence[-1] != END_TOKEN and len(sentence) < MAX_GEN_LEN:
        possible_next_words = df[df.iloc[:, :-1].eq(sentence[-n:]).all(axis=1)]
        if not possible_next_words.empty:
            next_word = possible_next_words['Next'].sample(1).iloc[0]
            sentence.append(next_word)
        else:
            break
    return ' '.join(sentence[n:-1]).capitalize() + '.'

# Streamlit UI
st.title('Small Language Model')

tabs = st.tabs(['Add Data', 'View Data', 'Frequencies', 'Generate Text'])

with tabs[0]:
    st.subheader('Add a New Sentence')
    user_input = st.text_input('Your sentence', placeholder='A child saw the dog.')
    if st.button('Submit'):
        add_training_data(user_input)
    st.text_area('Allowed Words:', ', '.join(sorted(ALLOWED_WORDS)), height=150)

with tabs[1]:
    st.subheader('View Existing Data')
    st.text('\n'.join(sentences))
    if st.button('Export Training Data'):
        pd.DataFrame(sentences, columns=['Sentences']).to_csv('training_data.csv')
        st.success('Training data exported to CSV.')

with tabs[2]:
    st.subheader('Word Frequencies')
    if not sentences:
        st.warning('Add training data first.')
    else:
        n = st.slider('Words used for prediction', 1, MAX_TUPLE_LEN, 1)
        df = get_tuples(n).value_counts().rename_axis(['Next', 'Previous']).reset_index(name='Frequency')
        st.dataframe(df)
        plot_frequencies(df.set_index(['Previous', 'Next']))

with tabs[3]:
    st.subheader('Generate Text')
    if not sentences:
        st.warning('Add training data first.')
    else:
        n = st.slider('Words used for prediction', 1, MAX_TUPLE_LEN, 1)
        df = get_tuples(n)
        if st.button('Generate Sentences'):
            sentences_generated = [generate_sentence(df) for _ in range(5)]
            st.text('\n'.join(sentences_generated))
            if st.button('Save Generated Sentences'):
                pd.DataFrame(sentences_generated, columns=['Sentences']).to_csv('generated_sentences.csv')
                st.success('Generated sentences exported to CSV.')
