import streamlit as st
from .snp import SNP500

@st.cache_resource
def fetch_data(start, end):
    snp500 = SNP500()
    snp500.get(start, end, progress=False)
    return snp500.data