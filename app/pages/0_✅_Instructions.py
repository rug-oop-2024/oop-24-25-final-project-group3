import streamlit as st
import pydoc

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)

st.markdown(open("INSTRUCTIONS.md").read())

if __name__ == "__main__":
    # Generate documentation for this module and save it as an HTML file
    pydoc.writedoc(__name__)
