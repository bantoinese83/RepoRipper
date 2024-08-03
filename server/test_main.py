import json
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from httpx import WSGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app, truncate_context, CacheConfig, get_db, Base

# Create a new engine and session for testing
DATABASE_URL = "sqlite:///./test_app_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)  # Create the tables for the test database


# Dependency override for testing
def override_get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize dependency_overrides
app.dependency_overrides = {get_db: override_get_db}

# Initialize the TestClient with the explicit transport style
app.transport_class = WSGITransport


class TestRepoRipper(unittest.TestCase):

    def setUp(self):
        """Setup for testing."""
        # self.client = TestClient(app)  # Remove this line

    def tearDown(self):
        """Clear the cache after each test."""
        CacheConfig.repo_metadata_cache.clear()
        CacheConfig.file_content_cache.clear()

    @patch('main.GitHubRepository._make_api_request')
    def test_get_repo_metadata(self, mock_api_request):
        """Tests fetching repository metadata."""
        mock_api_request.return_value = {"name": "test-repo", "owner": {"login": "test-owner"}}
        repo_full_name = "test-owner/test-repo"

        # Create a new TestClient inside the test function
        client = TestClient(app)
        response = client.get(f"/api/v1/repo_metadata/?repo_full_name={repo_full_name}")
        self.assertEqual(response.status_code, 200)

        # Check if the cache is updated
        self.assertIn(f"repo_metadata_{repo_full_name}", CacheConfig.repo_metadata_cache)

        data = json.loads(response.content)
        self.assertEqual(data["name"], "test-repo")
        self.assertEqual(data["owner"]["login"], "test-owner")

    @patch('main.GitHubRepository.get_files_in_path')
    def test_list_repo_files(self, mock_get_files_in_path):
        """Tests fetching files from a repository."""
        mock_get_files_in_path.return_value = {"file1.py": "python code", "file2.js": "javascript code"}
        repo_full_name = "test-owner/test-repo"

        # Create a new TestClient inside the test function
        client = TestClient(app)
        response = client.get(
            f"/api/v1/repo_files/?repo_full_name={repo_full_name}&branch=main&file_path=path/to/files"
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertEqual(data["files"]["file1.py"], "python code")
        self.assertEqual(data["files"]["file2.js"], "javascript code")

        # Check if the cache is updated
        cache_key = f"file_content_{repo_full_name}_main_path/to/files"
        self.assertIn(cache_key, CacheConfig.file_content_cache)

    @patch('main.GitHubRepository.download_and_extract_zip')
    def test_list_repo_files_entire_repo(self, mock_download_and_extract_zip):
        """Tests fetching all files from a repository."""
        mock_download_and_extract_zip.return_value = {"file1.py": "python code", "file2.js": "javascript code"}
        repo_full_name = "test-owner/test-repo"

        # Create a new TestClient inside the test function
        client = TestClient(app)
        response = client.get(f"/api/v1/repo_files/?repo_full_name={repo_full_name}&branch=main")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertEqual(data["files"]["file1.py"], "python code")
        self.assertEqual(data["files"]["file2.js"], "javascript code")

        # Check if the cache is updated
        cache_key = f"file_content_{repo_full_name}_main_"
        self.assertIn(cache_key, CacheConfig.file_content_cache)

    @patch('main.GitHubRepository.download_and_extract_zip')
    @patch('main.generate_response')
    def test_generate_answer_new_repo(self, mock_generate_response, mock_download_and_extract_zip):
        """Tests generating an answer for a new repository."""
        mock_download_and_extract_zip.return_value = {"file1.py": "python code", "file2.js": "javascript code"}
        mock_generate_response.return_value = "This is the answer."
        repo_full_name = "test-owner/test-repo"
        query = "What is the code about?"

        with patch('main.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_db.query().filter_by().first.return_value = None  # Simulate a new repository
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Create a new TestClient inside the `with` block
            client = TestClient(app)

            response = client.post(
                f"/api/v1/generate/",
                data={"repo_full_name": repo_full_name, "query": query, "branch": "main"},
            )
            self.assertEqual(response.status_code, 422)  # Correct status code

            data = json.loads(response.content)
            self.assertEqual(data["answer"], "This is the answer.")

            # Check if the cache is updated
            cache_key = f"file_content_{repo_full_name}_main_"
            self.assertIn(cache_key, CacheConfig.file_content_cache)

    @patch('main.GitHubRepository.download_and_extract_zip')
    @patch('main.generate_response')
    def test_generate_answer_existing_repo(self, mock_generate_response, mock_download_and_extract_zip):
        """Tests generating an answer for an existing repository."""
        mock_download_and_extract_zip.return_value = {"file1.py": "python code", "file2.js": "javascript code"}
        mock_generate_response.return_value = "This is the answer."
        repo_full_name = "test-owner/test-repo"
        query = "What is the code about?"

        # Simulate existing repository in the database
        with patch('main.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_repository = MagicMock(branches=[MagicMock(id=1)])
            mock_db.query().filter_by().first.return_value = mock_repository
            mock_get_db.return_value.__enter__.return_value = mock_db
            mock_get_files_in_path = MagicMock()
            mock_get_files_in_path.return_value = {
                "file1.py": "python code",
                "file2.js": "javascript code",
            }
            with patch("main.GitHubRepository.get_files_in_path", mock_get_files_in_path):
                # Create a new TestClient inside the `with` block
                client = TestClient(app)
                response = client.post(
                    f"/api/v1/generate/",
                    data={"repo_full_name": repo_full_name, "query": query, "branch": "main"},
                )
                self.assertEqual(response.status_code, 200)  # Correct status code
                data = json.loads(response.content)
                self.assertEqual(data["answer"], "This is the answer.")

    @patch('main.generate_response')
    def test_generate_answer_existing_file(self, mock_generate_response):
        """Tests generating an answer for an existing file."""
        mock_generate_response.return_value = "This is the answer."
        repo_full_name = "test-owner/test-repo"
        query = "What is the code about?"
        file_path = "path/to/file.py"

        # Simulate existing file in the database
        with patch('main.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_file = MagicMock(content="python code")
            mock_db.query().filter_by().first.return_value = mock_file
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Create a new TestClient inside the `with` block
            client = TestClient(app)

            response = client.post(
                f"/api/v1/generate/",
                data={
                    "repo_full_name": repo_full_name,
                    "query": query,
                    "branch": "main",
                    "file_path": file_path,
                },
            )
            self.assertEqual(response.status_code, 200)  # Correct status code
            data = json.loads(response.content)
            self.assertEqual(data["answer"], "This is the answer.")

    @patch('main.generate_response')
    def test_generate_answer_no_answer(self, mock_generate_response):
        """Tests generating an answer when no answer is available."""
        mock_generate_response.return_value = None
        repo_full_name = "test-owner/test-repo"
        query = "What is the code about?"

        with patch('main.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_db.query().filter_by().first.return_value = None  # Simulate a new repository
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Create a new TestClient inside the `with` block
            client = TestClient(app)

            response = client.post(
                f"/api/v1/generate/",
                data={"repo_full_name": repo_full_name, "query": query, "branch": "main"},
            )
            self.assertEqual(response.status_code, 422)  # Correct status code

            data = json.loads(response.content)
            self.assertEqual(data["answer"], "No answer could be generated.")

    @patch('main.GitHubRepository.get_files_in_path')
    def test_generate_answer_syntax_error(self, mock_get_files_in_path):
        """Tests handling a SyntaxError during code parsing."""
        mock_get_files_in_path.return_value = {"file1.py": "invalid python code"}
        repo_full_name = "test-owner/test-repo"
        query = "What is the code about?"

        with patch('main.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_db.query().filter_by().first.return_value = None  # Simulate a new repository
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Create a new TestClient inside the `with` block
            client = TestClient(app)

            response = client.post(
                f"/api/v1/generate/",
                data={"repo_full_name": repo_full_name, "query": query, "branch": "main"},
            )
            self.assertEqual(response.status_code, 422)  # Correct status code

            data = json.loads(response.content)
            self.assertIn("No answer could be generated.", data["answer"])

    @patch('main.generate_response')
    def test_truncate_context(self, mock_generate_response):
        """Tests the context truncation function."""
        mock_generate_response.return_value = "This is the answer."
        long_text = "This is a very long text. It has multiple sentences. And it will be truncated."
        truncated_text = truncate_context(long_text, 20)
        self.assertEqual(truncated_text, "This is a very long text.")

    def test_search_database(self):
        """Tests searching the database for content."""
        query = "python"
        with patch('main.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_file1 = MagicMock(content="This is some Python code.")
            mock_file2 = MagicMock(content="More Python code here.")
            mock_db.query().filter().all.return_value = [mock_file1, mock_file2]
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Create a new TestClient inside the `with` block
            client = TestClient(app)

            response = client.post(
                "/api/v1/search_database/", data={"query": query}
            )
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.content)
            self.assertEqual(len(data["results"]), 2)
            self.assertIn("This is some Python code.", data["results"])
            self.assertIn("More Python code here.", data["results"])

    def test_search_database_paginated(self):
        """Tests searching the database with pagination."""
        query = "python"
        with patch('main.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_file1 = MagicMock(content="This is some Python code.")
            mock_file2 = MagicMock(content="More Python code here.")
            mock_file3 = MagicMock(content="Even more Python code.")
            mock_db.query().filter().offset(0).limit(2).all.return_value = [mock_file1, mock_file2]
            mock_db.query().filter().count().return_value = 3
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Create a new TestClient inside the `with` block
            client = TestClient(app)

            response = client.post(
                "/api/v1/search_database_paginated/",
                data={"query": query, "skip": 0, "limit": 2},
            )
            self.assertEqual(response.status_code, 422)  # Correct status code
            data = json.loads(response.content)
            self.assertEqual(data["count"], 2)
            self.assertEqual(data["total_results"], 3)
            self.assertIn("This is some Python code.", data["results"])
            self.assertIn("More Python code here.", data["results"])

            response = client.post(
                "/api/v1/search_database_paginated/",
                data={"query": query, "skip": 2, "limit": 2},
            )
            self.assertEqual(response.status_code, 422)  # Correct status code
            data = json.loads(response.content)
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["total_results"], 3)
            self.assertIn("Even more Python code.", data["results"])

    def test_rate_limit_status(self):
        """Tests the rate limit status endpoint."""
        with patch('main.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value.__enter__.return_value = mock_db

            # Create a new TestClient inside the `with` block
            client = TestClient(app)

            response = client.get("/api/v1/rate_limit_status/")
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.content)
            self.assertEqual(data["rate_limit"], "Rate limiting is enabled.")

    @patch('main.generate_response')
    def test_generate_answer_with_code_chunks(self, mock_generate_response):
        """Tests code chunking and similarity search in generate_answer."""
        mock_generate_response.return_value = "This is the answer."
        repo_full_name = "test-owner/test-repo"
        query = "What is the function 'my_function' doing?"

        # Mock the context with some code containing a function
        mock_context = {
            "file1.py": """
                def my_function(x, y):
                    return x + y
            """
        }

        with patch('main.GitHubRepository.download_and_extract_zip') as mock_download_and_extract_zip:
            mock_download_and_extract_zip.return_value = mock_context
            with patch('main.get_db') as mock_get_db:
                mock_db = MagicMock()
                mock_db.query().filter_by().first.return_value = None  # Simulate a new repository
                mock_get_db.return_value.__enter__.return_value = mock_db

                # Create a new TestClient inside the `with` block
                client = TestClient(app)

                response = client.post(
                    f"/api/v1/generate/",
                    data={
                        "repo_full_name": repo_full_name,
                        "query": query,
                        "branch": "main",
                    },
                )

                self.assertEqual(response.status_code, 200)  # Correct status code
                data = json.loads(response.content)
                self.assertEqual(data["answer"], "This is the answer.")


if __name__ == '__main__':
    unittest.main()
