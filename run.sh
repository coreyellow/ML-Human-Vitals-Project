#!/bin/bash
cd "$(dirname "$0")"
/usr/local/bin/python3 -m streamlit run UI/app.py --server.headless true
