import streamlit as st
import pydoc

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.sidebar.success("Select a page above.")
st.markdown(open("README.md").read())


if __name__ == "__main__":
    # Generate documentation for this module and save it as an HTML file
    pydoc.writedoc(__name__)
