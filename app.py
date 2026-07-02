
import streamlit as st
import joblib
import pandas as pd
import sqlite3
import hashlib
import secrets
from typing import Optional, Tuple


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hash_val = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return salt, hash_val

def verify_password(stored_salt: str, stored_hash: str, password_attempt: str) -> bool:
    _, h = hash_password(password_attempt, salt=stored_salt)
    return h == stored_hash


@st.cache_resource
def get_connection():
    # returns a persistent connection cached by Streamlit
    return sqlite3.connect("predictions.db", check_same_thread=False)

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    # users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        salt TEXT,
        role TEXT DEFAULT 'user'
    )
    """)
    # history table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        year INTEGER,
        area REAL,
        fertilizer REAL,
        pesticide REAL,
        state TEXT,
        season TEXT,
        crop TEXT,
        yield_pred REAL,
        prod_pred REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    conn.commit()
    ensure_admin(conn)

def ensure_admin(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    if cur.fetchone() is None:
        salt, pw_hash = hash_password("admin123")
        cur.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
            ("admin", pw_hash, salt, "admin"),
        )
        conn.commit()

def create_user(conn: sqlite3.Connection, username: str, password: str, role: str = "user") -> bool:
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cur.fetchone() is not None:
        return False
    salt, pw_hash = hash_password(password)
    cur.execute(
        "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
        (username, pw_hash, salt, role),
    )
    conn.commit()
    return True

def authenticate_user(conn: sqlite3.Connection, username: str, password: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash, salt, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        return None
    user_id, uname, pw_hash, salt, role = row
    if verify_password(salt, pw_hash, password):
        return {"id": user_id, "username": uname, "role": role}
    return None

def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT id, username, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "role": row[2]}

def get_all_users(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT id, username, role FROM users", conn)

def delete_user(conn: sqlite3.Connection, user_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()

def insert_history(conn: sqlite3.Connection, user_id: Optional[int], year: int, area: float, fertilizer: float,
                   pesticide: float, state: str, season: str, crop: str, y_pred: float, p_pred: float):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history
        (user_id, year, area, fertilizer, pesticide, state, season, crop, yield_pred, prod_pred)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, year, area, fertilizer, pesticide, state, season, crop, y_pred, p_pred))
    conn.commit()

def fetch_history_for_user(conn: sqlite3.Connection, user_id: int) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM history WHERE user_id = ? ORDER BY id DESC", conn, params=(user_id,))

def fetch_all_history(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)

def delete_history_entry(conn: sqlite3.Connection, entry_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM history WHERE id = ?", (entry_id,))
    conn.commit()

def update_user_role(conn: sqlite3.Connection, user_id: int, new_role: str):
    cur = conn.cursor()
    cur.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    conn.commit()

# ------------------------
# Load ML artifacts (make sure these files exist)
# ------------------------
try:
    model_yield = joblib.load("model_yield.pkl")
    model_prod = joblib.load("model_prod.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")  # list of X.columns used in training
except Exception as e:
    st.error("Error loading ML artifacts. Make sure model_yield.pkl, model_prod.pkl, scaler.pkl, columns.pkl exist in the app folder.")
    st.stop()

# ------------------------
# App UI and logic
# ------------------------
st.set_page_config(page_title="Crop Yield Predictor ", layout="wide")
st.title("🌾 Crop Yield & Production Predictor ")

# initialize DB / connection
conn = get_connection()
init_db(conn)

# prepare dropdown lists (you can modify these lists as per your dataset)
indian_states = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka",
    "Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram",
    "Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
    "Tripura","Uttar Pradesh","Uttarakhand","West Bengal"
]
seasons_list = ["Kharif", "Rabi", "Whole Year", "Autumn", "Summer", "Winter"]
crops_list = [
    "Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Barley", "Oilseeds",
    "Pulses", "Jowar", "Bajra", "Groundnut", "Sunflower", "Soybean",
    "Tur", "Urad", "Moong", "Sesamum", "Jute", "Tea", "Coffee"
]

# ------------------------
# Session & initial role-select
# ------------------------
if "role_choice" not in st.session_state:
    st.session_state.role_choice = None  # 'admin' or 'farmer' or None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# Role selection landing (show only if role not chosen and not logged-in)
if st.session_state.role_choice is None and not st.session_state.logged_in:
    st.markdown("## 👋 Welcome")
    st.info("Please select your role to continue.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("I am Admin"):
            st.session_state.role_choice = "admin"
            st.rerun()

    with col2:
        if st.button("I am Farmer"):
            st.session_state.role_choice = "farmer"
            st.rerun()

    st.markdown("---")
    st.write("If you've already used the app, choose your role above and continue to login/signup.")
    st.stop()

# ------------------------
# Sidebar common (shows after role chosen or logged-in)
# ------------------------
with st.sidebar:
    st.header("Account")
    if st.session_state.logged_in:
        st.write(f"Logged in as: **{st.session_state.user['username']}**")
        st.write(f"Role: **{st.session_state.user['role']}**")
        if st.button("Logout"):
            # clear session and go back to role selection
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.role_choice = None
            st.rerun()
    else:
        st.write(f"Role selected: **{st.session_state.role_choice}**")
        st.info("Please login or signup to use the app.")
    st.markdown("---")
   

# Decide pages shown in sidebar once logged in
pages = []
if not st.session_state.logged_in:
    pages = ["Login", "Signup"]
else:
    # logged in: show appropriate pages
    pages = ["Predict", "My History"]
    if st.session_state.user and st.session_state.user.get("role") == "admin":
        pages = ["Admin Dashboard"]

page = st.sidebar.selectbox("Navigate", pages)

# ------------------------
# AUTH PAGES
# ------------------------
if page == "Login":
    st.subheader("🔐 Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")

    if st.button("Login"):
        auth = authenticate_user(conn, username_input.strip(), password_input.strip())
        if auth:
            # role check: ensure user matches chosen role
            if st.session_state.role_choice == "admin" and auth["role"] != "admin":
                st.error("Access denied. You selected Admin but credentials are not admin.")
            elif st.session_state.role_choice == "farmer" and auth["role"] != "user":
                st.error("Access denied. You selected Farmer but credentials are not a farmer user.")
            else:
                st.session_state.logged_in = True
                st.session_state.user = auth
                st.success("Login successful.")
                st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown("---")
    st.write("Don't have an account? Signup below.")
    if st.button("Go to Signup"):
        st.session_state.role_choice = st.session_state.role_choice  # keep role
        st.sidebar.selectbox("Navigate", ["Signup"])  # no-op to help UI
        st.rerun()

elif page == "Signup":
    st.subheader("📝 Signup")
    st.write(f"Create a {st.session_state.role_choice} account")
    new_user = st.text_input("Choose a username")
    new_pass = st.text_input("Choose a password", type="password")
    confirm_pass = st.text_input("Confirm password", type="password")

    if st.button("Signup"):
        if not new_user or not new_pass:
            st.error("Provide username and password.")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match.")
        else:
            role_to_create = "admin" if st.session_state.role_choice == "admin" else "user"
            # Prevent creating second admin via signup page (admin should be only default)
            if role_to_create == "admin":
                st.error("Admin account cannot be created here.")
            else:
                ok = create_user(conn, new_user.strip(), new_pass.strip(), role=role_to_create)
                if ok:
                    st.success("Account created. Please login from Login page.")
                else:
                    st.error("Username already exists. Choose another.")

# ------------------------
# PREDICTION PAGE (Farmer)
# ------------------------
elif page == "Predict":
    if not st.session_state.logged_in:
        st.error("Login required to make predictions.")
    else:
        # only users (farmers) should use this page; admin redirected
        if st.session_state.user.get("role") == "admin":
            st.error("Admins cannot use prediction page here.")
        else:
            st.subheader("🔮 Make a Prediction")
            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.number_input("Year", 1990, 2030, value=2020)
                area = st.number_input("Area (hectares)", min_value=0.0, value=1000.0)
            with col2:
                fertilizer = st.number_input("Fertilizer usage", min_value=0.0, value=500.0)
                pesticide = st.number_input("Pesticide usage", min_value=0.0, value=200.0)
            with col3:
                state = st.selectbox("State", indian_states)
                season = st.selectbox("Season", seasons_list)
                crop = st.selectbox("Crop", crops_list)

            if st.button("Predict Now"):
                new_df = pd.DataFrame({
                    "year": [year],
                    "area": [area],
                    "fertilizer": [fertilizer],
                    "pesticide": [pesticide],
                    "state": [state],
                    "season": [season],
                    "crop": [crop]
                })
                new_enc = pd.get_dummies(new_df, drop_first=True)
                for c in columns:
                    if c not in new_enc.columns:
                        new_enc[c] = 0
                new_enc = new_enc[columns]
                new_scaled = scaler.transform(new_enc)
                pred_y = model_yield.predict(new_scaled)[0]
                pred_p = model_prod.predict(new_scaled)[0]

                st.success(f"🌱 Predicted Yield: {pred_y:.2f} kg/ha")
                st.success(f"🚜 Predicted Production: {pred_p:.2f} metric tonnes")

                # save into DB for this user
                uid = st.session_state.user["id"]
                insert_history(conn, uid, year, area, fertilizer, pesticide, state, season, crop, pred_y, pred_p)
                st.info("Prediction saved to your history.")

# ------------------------
# MY HISTORY (Farmer)
# ------------------------
elif page == "My History":
    if not st.session_state.logged_in:
        st.error("Login required.")
    else:
        if st.session_state.user.get("role") == "admin":
            st.error("Admins do not have personal history here.")
        else:
            st.subheader("📜 My Prediction History")
            uid = st.session_state.user["id"]
            df_user = fetch_history_for_user(conn, uid)
            st.dataframe(df_user)

            if not df_user.empty:
                st.markdown("**Delete an entry (by ID)**")
                del_id = st.number_input("ID to delete", min_value=1, step=1)
                if st.button("Delete My Entry"):
                    cur = conn.cursor()
                    cur.execute("SELECT user_id FROM history WHERE id = ?", (del_id,))
                    row = cur.fetchone()
                    if not row:
                        st.error("Entry not found.")
                    elif row[0] != uid:
                        st.error("You can only delete your own entries.")
                    else:
                        delete_history_entry(conn, del_id)
                        st.success("Deleted.")
                        st.rerun()

# ------------------------
# ADMIN DASHBOARD
# ------------------------
elif page == "Admin Dashboard":
    if not st.session_state.logged_in or st.session_state.user.get("role") != "admin":
        st.error("Admin access only.")
    else:
        st.subheader("🛠 Admin Dashboard")
        admin_tabs = st.tabs(["All Users", "All Predictions", "Manage Users"])

        # All Users
        with admin_tabs[0]:
            st.markdown("### 👥 Registered Users")
            df_users = get_all_users(conn)
            st.dataframe(df_users)

        # All Predictions
        with admin_tabs[1]:
            st.markdown("### 📦 All Predictions")
            df_all = fetch_all_history(conn)
            st.dataframe(df_all)

            st.markdown("**Delete any prediction (by ID)**")
            del_pred = st.number_input("Prediction ID to delete", min_value=1, step=1, key="del_pred")
            if st.button("Delete Prediction (Admin)"):
                delete_history_entry(conn, del_pred)
                st.success("Deleted prediction.")
                st.rerun()

        # Manage Users (roles / delete)
        with admin_tabs[2]:
            st.markdown("### Manage Users")
            df_users = get_all_users(conn)
            st.dataframe(df_users)

            st.markdown("Change role for a user")
            user_to_change = st.text_input("Username to change role")
            new_role = st.selectbox("New role", ["user", "admin"])
            if st.button("Change Role"):
                u = get_user_by_username(conn, user_to_change.strip())
                if not u:
                    st.error("User not found.")
                elif u["username"] == "admin":
                    st.error("Default admin role cannot be changed here.")
                else:
                    update_user_role(conn, u["id"], new_role)
                    st.success("Role updated.")
                    st.rerun()

            st.markdown("Delete a user (this will also delete their history)")
            del_user_name = st.text_input("Username to delete")
            if st.button("Delete User"):
                u = get_user_by_username(conn, del_user_name.strip())
                if not u:
                    st.error("User not found.")
                elif u["username"] == "admin":
                    st.error("Cannot delete default admin.")
                else:
                    delete_user(conn, u["id"])
                    st.success("User deleted.")
                    st.rerun()
