import streamlit as st
import pandas as pd
import hashlib
import os
import requests
import streamlit_lottie

# Set the page to default to wide mode
st.set_page_config(page_title="PropIntel", page_icon="🏠", layout="wide")

def load_lottieur(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# CSV file paths
CUSTOMER_FILE = "/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/customer_data.csv"
ADMIN_FILE = "/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/admin_owner_data.csv"

def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_role"] = None
        st.session_state["user_name"] = None
        st.session_state["page"] = "Login"  # Default page

    def initialize_csv(file_path):
        """Ensures CSV file exists and has proper headers."""
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            df = pd.DataFrame(columns=["name", "email", "password"])
            df.to_csv(file_path, index=False)

    initialize_csv(CUSTOMER_FILE)
    initialize_csv(ADMIN_FILE)

    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    def load_user_data(file_path):
        """Loads user data from CSV and ensures proper headers exist."""
        if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
            return pd.read_csv(file_path)
        else:
            return pd.DataFrame(columns=["name", "email", "password"])  # ✅ Safe empty DataFrame

    def check_login(file_path, identifier, password):
        df = load_user_data(file_path)
        hashed_password = hash_password(password)

        if identifier in df["email"].values or identifier in df["name"].values:
            user_row = df[(df["email"] == identifier) | (df["name"] == identifier)]
            if not user_row.empty and user_row["password"].values[0] == hashed_password:
                return True, user_row["name"].values[0]
        return False, None

    # 🚀 If logged in, switch to the respective page
    if st.session_state["logged_in"]:
        if st.session_state["user_role"] == "Customer":
            st.switch_page("/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/pages/customer.py")  # ✅ Redirect to Customer Dashboard
        elif st.session_state["user_role"] == "Admin Owner":
            st.switch_page("/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/pages/admin.py")  # ✅ Redirect to Admin Dashboard

        # 🔹 Hide Sidebar Before Login
    if not st.session_state["logged_in"]:
        st.markdown("""
            <style>
            [data-testid="stSidebar"] {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True)
    
        # Center align using HTML
        st.markdown(
            """
            <h1 style='text-align: center;'>PropIntel</h1>
            <h3 style='text-align: center;'>Smart Insights, Smarter Investments.</h3>
            """,
            unsafe_allow_html=True
        )

    st.title(" ")

    col1, col2 = st.columns(2)
    with col1:
        l1 = "https://lottie.host/5ae01eac-69f3-4f8b-8703-137ca5bbfc31/cSY7ipxxfn.json"
        st.lottie(l1)
    with col2:
        # Display login or signup form based on session state
        if st.session_state["page"] == "Login":
            st.subheader("Login to Your Account")
            identifier = st.text_input("Enter your name or email")
            password = st.text_input("Enter your password", type="password")
            role = st.radio("Select Role", ["Customer", "Admin Owner"], horizontal=True)
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("Login", use_container_width=True):
                    file_path = CUSTOMER_FILE if role == "Customer" else ADMIN_FILE
                    login_success, user_name = check_login(file_path, identifier, password)

                    if login_success:
                        st.session_state["logged_in"] = True
                        st.session_state["user_role"] = role  # ✅ Set user role correctly
                        st.session_state["user_name"] = user_name
                        st.success(f"Login successful! Redirecting to {role} page...")

                        # ✅ Redirect after successful login
                        if role == "Customer":
                            st.switch_page("/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/pages/customer.py")
                        elif role == "Admin Owner":
                            st.switch_page("/Users/jsujanchowdary/Downloads/langchain-ask-csv-main/pages/admin.py")
                    else:
                        st.error("Invalid credentials! Please check your details.")
            with col4:
                # Button to switch to Sign-Up form
                if st.button("Create a New Account", use_container_width=True):
                    st.session_state["page"] = "Sign Up"
                    st.rerun()

        elif st.session_state["page"] == "Sign Up":
            st.subheader("Create a New Account (Customers Only)")
            name = st.text_input("Enter your name")
            email = st.text_input("Enter your email")
            password = st.text_input("Enter your password", type="password")
            re_password = st.text_input("Re-enter your password", type="password")

            col5, col6 = st.columns(2)

            with col5:
                if st.button("Sign Up", use_container_width=True):
                    if password != re_password:
                        st.error("Passwords do not match!")
                    else:
                        df = load_user_data(CUSTOMER_FILE)
                        if email in df["email"].values:
                            st.warning("This email is already registered!")
                        else:
                            new_user = pd.DataFrame([[name, email, hash_password(password)]], 
                                                    columns=["name", "email", "password"])
                            df = pd.concat([df, new_user], ignore_index=True)
                            df.to_csv(CUSTOMER_FILE, index=False)
                            st.success("Sign up successful! You can now log in.")

                            # Switch back to login page
                            st.session_state["page"] = "Login"
                            st.rerun()

            with col6:
                # Button to switch back to Login form
                if st.button("Back to Login", use_container_width=True):
                    st.session_state["page"] = "Login"
                    st.rerun()

if __name__ == "__main__":
    main()