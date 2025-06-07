
# RAG_Chatbot

## Overview

**RAG_Chatbot** is a Python-based chatbot application that leverages Retrieval-Augmented Generation (RAG) to provide contextually relevant responses. By integrating a local SQLite database, the chatbot retrieves pertinent information to enhance its conversational capabilities.

## Features

- **Retrieval-Augmented Generation (RAG):** Combines information retrieval with generative models to produce informed responses.
- **SQLite Integration:** Utilizes a local SQLite database (`our_sql_database.db`) to store and retrieve contextual data.
- **Modular Design:** Structured codebase with clear separation of concerns, facilitating maintenance and scalability.

## Project Structure

```
RAG_Chatbot/
├── app.py                 # Main application script
├── our_sql_database.db    # SQLite database file
├── requirements.txt       # Python dependencies
└── runtime.txt            # Runtime environment specification
```

## Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)

## Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/FadyHassanein/RAG_Chatbot.git
cd RAG_Chatbot
```

2. **Create a Virtual Environment (Optional but Recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

1. **Ensure the SQLite Database Exists:**

The application relies on `our_sql_database.db`. Ensure this file is present in the project root.

2. **Run the Application:**

```bash
python app.py
```

3. **Interact with the Chatbot:**

Once running, the chatbot will prompt for user input. Enter queries or messages, and the chatbot will respond using RAG techniques.

## Configuration

- **Database Schema:** The structure of `our_sql_database.db` should align with the application's expectations.
- **Runtime Environment:** The `runtime.txt` file specifies the Python runtime version.

## Contributing

1. **Fork the Repository**
2. **Create a New Branch:**

```bash
git checkout -b feature/your-feature-name
```

3. **Commit Your Changes:**

```bash
git commit -m "Add your feature"
```

4. **Push to Your Fork:**

```bash
git push origin feature/your-feature-name
```

5. **Create a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue on the [GitHub repository](https://github.com/FadyHassanein/RAG_Chatbot/issues).
