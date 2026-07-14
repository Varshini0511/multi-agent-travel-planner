"""
One place to resolve secrets, working in every environment the app runs in:

  - Local dev / `python main.py`: values come from a .env file (loaded here
    via python-dotenv) into os.environ.
  - Streamlit Community Cloud: there is NO .env and secrets are NOT set as
    environment variables - they're only reachable through st.secrets, which
    is populated from the app's Secrets box in the dashboard.

get_secret() checks os.environ first, then falls back to st.secrets, so the
rest of the code (app/db.py, app/llm.py, app/tools.py) never has to care which
environment it's running in.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def get_secret(key: str, required: bool = True) -> str | None:
    value = os.environ.get(key)

    if not value:
        # Only reachable under a Streamlit runtime; harmless (caught) otherwise.
        try:
            import streamlit as st

            value = st.secrets.get(key)
        except Exception:
            value = None

    if not value and required:
        raise KeyError(
            f"Missing required secret '{key}'. Set it in .env locally, or in "
            f"the Streamlit Cloud app's Settings -> Secrets."
        )
    return value
