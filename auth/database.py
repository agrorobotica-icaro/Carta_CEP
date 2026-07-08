import sqlite3
from pathlib import Path
import bcrypt

# Caminho do banco
DB_PATH = Path(__file__).parent / "users.db"


def get_connection():
    """
    Retorna uma conexão com o banco SQLite.
    """
    return sqlite3.connect(DB_PATH)


def initialize_database():
    """
    Cria o banco de dados e a tabela de usuários caso não existam.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            active INTEGER DEFAULT 1
        )
    """)

    conn.commit()
    create_default_admin()
    conn.close()
def create_default_admin():
    """
    Cria o usuário administrador padrão caso ele ainda não exista.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    )

    if cursor.fetchone() is None:

        password = bcrypt.hashpw(
            "admin123".encode(),
            bcrypt.gensalt()
        ).decode()

        cursor.execute("""
            INSERT INTO users
            (username, password, name, role)
            VALUES (?, ?, ?, ?)
        """, (
            "admin",
            password,
            "Administrador",
            "Admin"
        ))

        conn.commit()

    conn.close()