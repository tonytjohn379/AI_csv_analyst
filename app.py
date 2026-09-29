
import os
import re
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai

import time
from google.genai import errors
from pathlib import Path
load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Data Analyst")
st.caption(
    "Upload a CSV, ask questions in plain English, "
    "and explore your data with AI."
)

# -------------------------------
# Gemini configuration
# -------------------------------
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

api_key = os.getenv("GEMINI_API_KEY", "").strip()

client = genai.Client(api_key=api_key) if api_key else None
MODEL = "gemini-flash-lite-latest"

def generate_ai_text(prompt):
    """Generate text using Gemini with retry handling."""

    if client is None:
        raise ValueError(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            return response.text or ""

        except errors.ServerError as exc:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError("Gemini request failed after retries.")




def dataframe_schema(df):
    """Describe the uploaded dataset for the AI."""
    details = []

    for column in df.columns:
        details.append(
            f"- {column!r}: {df[column].dtype}"
        )

    return "\n".join(details)


def generate_sql(question, df):
    """Ask Gemini to generate a read-only SQL query."""
    schema = dataframe_schema(df)

    prompt = f"""
You are an expert SQLite data analyst.

The uploaded CSV is stored in a SQLite table named uploaded_data.

Table schema:
{schema}

User question:
{question}

Rules:
1. Return exactly one SQLite SELECT query.
2. A WITH clause followed by SELECT is allowed.
3. Use only the uploaded_data table.
4. Use double quotes around column names when needed.
5. Never generate INSERT, UPDATE, DELETE, DROP, ALTER,
   CREATE, ATTACH, PRAGMA, or other modifying statements.
6. Do not use Markdown code fences.
7. Return only the SQL query.
"""

    return generate_ai_text(prompt).strip()


def validate_sql(sql):
    """Reject anything other than a single read-only query."""
    cleaned = sql.strip()

    # Remove optional Markdown fences if the model adds them.
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Reject multiple statements.
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed.")

    # Only SELECT or WITH queries are accepted.
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    # Block dangerous operations and database metadata access.
    forbidden = (
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
        r"ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|"
        r"LOAD_EXTENSION|INSTALL|COPY)\b"
    )

    if re.search(forbidden, cleaned, re.IGNORECASE):
        raise ValueError("The SQL contains a prohibited operation.")

    if re.search(r"\bsqlite_master\b|\bsqlite_schema\b", cleaned,
                 re.IGNORECASE):
        raise ValueError("Database metadata queries are not allowed.")

    if not re.search(
    r'\bFROM\s+["`\[]?uploaded_data["`\]]?\b',
    cleaned,
    re.IGNORECASE
    ):
        raise ValueError(
        "The query must read from the uploaded_data table."
        )
    return cleaned


# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("The uploaded CSV has no data rows.")
            st.stop()

        if len(df.columns) == 0:
            st.error("The CSV does not contain any columns.")
            st.stop()

        if len(df.columns) > 100:
            st.error("Please upload a file with at most 100 columns.")
            st.stop()

        if not df.columns.is_unique:
            st.error("Column names must be unique.")
            st.stop()

    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

    # Display dataset overview.
    st.success(f"Loaded: {uploaded_file.name}")

    total_rows, total_columns = df.shape

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows", f"{total_rows:,}")
    col2.metric("Total columns", total_columns)
    col3.metric("Missing cells", int(df.isna().sum().sum()))
    col4.metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Column information"):
        profile = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(t) for t in df.dtypes],
            "Missing values": df.isna().sum().values,
            "Unique values": df.nunique().values
        })
        st.dataframe(profile, use_container_width=True)

    with st.expander("Missing values by column"):
        missing = df.isna().sum().rename("Missing values")
        missing = missing[missing > 0]
        if missing.empty:
            st.write("No missing values found.")
        else:
            st.dataframe(missing.to_frame())

    # -------------------------------
    # Natural-language analysis
    # -------------------------------
    st.divider()
    st.subheader("🤖 Ask your data")

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: Which product generated the highest revenue?"
        )
    )

    if st.button("Analyze data", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            with st.spinner("Generating SQL with AI..."):
                sql = generate_sql(question, df)

            st.subheader("Generated SQL")
            st.code(sql, language="sql")

            with st.spinner("Validating SQL..."):
                sql = validate_sql(sql)

            # Create an isolated in-memory SQLite database.
            connection = sqlite3.connect(":memory:")

            try:
                df.to_sql(
                    "uploaded_data",
                    connection,
                    index=False,
                    if_exists="replace"
                )

                with st.spinner("Executing SQL..."):
                    result = pd.read_sql_query(sql, connection)

            finally:
                connection.close()

            st.subheader("Query results")
            st.dataframe(result, use_container_width=True)

            st.download_button(
                "Download results as CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="analysis_results.csv",
                mime="text/csv"
            )

            # Automatically suggest a chart when possible.
            numeric_columns = result.select_dtypes(
                include="number"
            ).columns.tolist()

            categorical_columns = [
                col for col in result.columns
                if col not in numeric_columns
            ]

            st.subheader("Interactive visualization")

            if not result.empty and numeric_columns:
                if categorical_columns:
                    x_column = categorical_columns[0]
                    y_column = numeric_columns[0]

                    if result[x_column].nunique() <= 30:
                        fig = px.bar(
                            result,
                            x=x_column,
                            y=y_column,
                            title=f"{y_column} by {x_column}"
                        )
                        st.plotly_chart(
                            fig, use_container_width=True
                        )
                    else:
                        st.info(
                            "Too many categories for an automatic bar chart."
                        )
                else:
                    fig = px.bar(
                        result,
                        y=numeric_columns[0],
                        title=numeric_columns[0]
                    )
                    st.plotly_chart(
                        fig, use_container_width=True
                    )
            else:
                st.info(
                    "No numeric result column was found for an "
                    "automatic chart."
                )

            # Ask AI to explain the actual query output.
            st.subheader("💡 AI-generated insights")

            preview = result.head(30).to_csv(index=False)

            insight_prompt = f"""
You are a data analyst.

User's question:
{question}

Executed SQL:
{sql}

Actual query results (CSV format, first 30 rows maximum):
{preview}

Result row count: {len(result)}

Explain the results clearly.
Give useful observations supported by the displayed data.
Do not invent numbers or claim causation without evidence.
Mention if the result is empty or insufficient for a conclusion.
"""

            with st.spinner("Generating insights..."):
                insights = generate_ai_text(insight_prompt)

            st.markdown(insights)

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption(
                "Check the question, generated SQL, API key, "
                "and uploaded column names before retrying."
            )

else:
    st.info("Upload a CSV file to get started.")
    st.markdown(
        """
        **Example questions**
        - How many rows are in each category?
        - What is the average price by city?
        - Which five products have the highest revenue?
        - Show the total sales for each month.
        - Which columns contain missing values?
        """
    )