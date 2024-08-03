# main.py
import ast
import io
import logging
import os
import zipfile
from typing import List, Optional, Dict

import absl.logging
import aiohttp
import numpy as np
import torch
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Form,
    Depends,
    Query,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google.auth.transport import requests
from google.generativeai import GenerativeModel, configure, GenerationConfig
from google.oauth2 import id_token
from halo import Halo
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import ValidationError, BaseModel
from sentence_transformers import SentenceTransformer, util
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Boolean,
    inspect,
    ForeignKey,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

from core.config import RAGConfig, CacheConfig

# --- Logging Configuration ---
absl.logging.set_verbosity("info")
absl.logging.use_absl_handler()
logging.basicConfig(level=logging.INFO)

# --- Global Configuration ---
DATABASE_URL = "sqlite:///./app_data.db"
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
API_V1_PREFIX = "/api/v1"
MAX_CONTEXT_LENGTH = 4000  # Adjust based on Gemini's limits


# --- Custom Exceptions ---
class RepositoryNotFound(HTTPException):
    def __init__(self, repo_full_name: str):
        super().__init__(
            status_code=404, detail=f"Repository '{repo_full_name}' not found."
        )


class InvalidCredentials(HTTPException):
    def __init__(self):
        super().__init__(status_code=401, detail="Incorrect username or password")


class GitHubAPIError(HTTPException):
    def __init__(self, status_code: int, message: str):
        super().__init__(
            status_code=status_code,
            detail=f"GitHub API Error: {message}",
        )


# --- Dependency Injection for Database and Other Services ---


class RateLimiter:
    def __init__(self):
        self.limiter = Limiter(key_func=get_remote_address)

    def limit(self, rate: str):
        return self.limiter.limit(rate)


def get_db():
    """Dependency function to provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_rate_limiter():
    """Dependency function to provide the rate limiter."""
    return RateLimiter()


def get_github_repository(repo_full_name: str):
    """Dependency function to provide a GitHubRepository instance."""
    return GitHubRepository(repo_full_name)


# --- Database Setup and Models ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)


class Repository(Base):
    __tablename__ = "repositories"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, unique=True, index=True)
    branches = relationship("Branch", back_populates="repository")


class Branch(Base):
    __tablename__ = "branches"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    repository = relationship("Repository", back_populates="branches")
    files = relationship("File", back_populates="branch")  # Add this line


class File(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    path = Column(String, index=True)
    content = Column(Text)  # Store the entire file content
    branch_id = Column(Integer, ForeignKey("branches.id"), nullable=False)
    branch = relationship("Branch", back_populates="files")


Base.metadata.create_all(bind=engine)


# --- GitHub Interaction ---


class GitHubRepository:
    def __init__(self, repo_full_name: str):
        self.repo_full_name = repo_full_name
        self.api_base_url = "https://api.github.com/repos/"

    async def _make_api_request(
            self, endpoint: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        """Makes an asynchronous API request to GitHub."""
        api_url = f"{self.api_base_url}{self.repo_full_name}{endpoint}"
        headers = {"Accept": "application/vnd.github+json"}
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(api_url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif 400 <= response.status < 500:
                        error_json = await response.json()
                        message = error_json.get(
                            "message", "Client error."
                        )  # Get an error message from response
                        raise GitHubAPIError(
                            status_code=response.status, message=message
                        )
                    else:
                        logging.warning(
                            f"GitHub API Request Failed: {response.status} - {api_url}"
                        )
                        raise GitHubAPIError(
                            status_code=response.status,
                            message="Failed to fetch repository data.",
                        )
        except aiohttp.ClientError as e:
            logging.error(f"HTTP request error: {e}")
            raise GitHubAPIError(
                status_code=500, message="Failed to connect to GitHub API."
            )

    async def get_branches(self) -> List[str]:
        """Fetches the list of branches for the repository."""
        branches_data = await self._make_api_request("/branches")
        return [branch["name"] for branch in branches_data] if branches_data else []

    async def get_repo_metadata(self) -> Optional[dict]:
        """Fetches the repository metadata."""
        cache_key = f"repo_metadata_{self.repo_full_name}"
        if cache_key in CacheConfig.repo_metadata_cache:  # Check the cache
            return CacheConfig.repo_metadata_cache[cache_key]

        metadata = await self._make_api_request("")
        if metadata:
            CacheConfig.repo_metadata_cache[cache_key] = metadata  # Store in the cache
        return metadata

    async def get_files_in_path(
            self, branch: str = "main", path: str = ""
    ) -> Dict[str, str]:
        """Fetches files within a specific path and branch."""
        cache_key = f"file_content_{self.repo_full_name}_{branch}_{path}"
        if cache_key in CacheConfig.file_content_cache:  # Check the cache
            return CacheConfig.file_content_cache[cache_key]

        contents_data = await self._make_api_request(
            f"/contents/{path}", params={"ref": branch}
        )
        if contents_data is None:
            return {}
        file_contents = {}
        valid_extensions = self._get_valid_extensions()
        for item in contents_data:
            if item["type"] == "file" and any(item["name"].endswith(ext) for ext in valid_extensions):
                file_contents[item["path"]] = await self.get_file_content(
                    item["download_url"]
                )
        CacheConfig.file_content_cache[cache_key] = file_contents
        return file_contents

    @staticmethod
    async def get_file_content(download_url: str) -> str:
        """Fetches the content of a single file."""
        async with aiohttp.ClientSession() as session:
            async with session.get(download_url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logging.warning(
                        f"Failed to download file from {download_url}. Status code: {response.status}"
                    )
                    return ""

    async def download_and_extract_zip(
            self, branch: str = "main", path: Optional[str] = None
    ) -> Dict[str, str]:
        """Downloads and extracts the ZIP file of the repository or a specific path."""
        cache_key = f"file_content_{self.repo_full_name}_{branch}_{path}"
        if cache_key in CacheConfig.file_content_cache:
            return CacheConfig.file_content_cache[cache_key]

        zip_url = (
            f"https://github.com/{self.repo_full_name}/archive/refs/heads/{branch}.zip"
        )
        extract_path = self.repo_full_name.split("/")[-1] if path else ""

        file_contents = {}
        spinner = Halo(text="Downloading and extracting repository...", spinner="dots")
        try:
            spinner.start()
            async with aiohttp.ClientSession() as session:
                async with session.get(zip_url) as response:
                    if response.status == 200:
                        zip_data = await response.read()
                        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                            for file_info in z.infolist():
                                file_name = file_info.filename
                                if extract_path:
                                    if (
                                            file_name.startswith(f"{extract_path}/")
                                            and not file_info.is_dir()
                                    ):
                                        file_name = file_name.replace(
                                            f"{extract_path}/", ""
                                        )
                                        if self._is_valid_extension(file_name):
                                            with z.open(file_info.filename) as f:
                                                file_contents[file_name] = (
                                                    f.read().decode("utf-8")
                                                )
                                else:
                                    if (
                                            not file_info.is_dir()
                                            and self._is_valid_extension(file_name)
                                    ):
                                        with z.open(file_info.filename) as f:
                                            file_contents[file_name] = f.read().decode(
                                                "utf-8"
                                            )
                    else:
                        logging.warning(
                            f"Failed to download ZIP file from {zip_url}. Status code: {response.status}"
                        )
        except aiohttp.ClientError as e:
            logging.error(f"HTTP request error: {e}")
        except zipfile.BadZipFile as e:
            logging.error(f"Failed to unzip file: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        finally:
            spinner.stop()

        CacheConfig.file_content_cache[cache_key] = file_contents
        return file_contents

    def _is_valid_extension(self, file_name: str) -> bool:
        """Checks if the file has a valid extension."""
        valid_extensions = self._get_valid_extensions()
        return any(file_name.endswith(ext) for ext in valid_extensions)

    @staticmethod
    def _get_valid_extensions() -> list:
        """Reads valid extensions from file_extensions.md."""
        with open("core/file_extensions.md", "r") as file:
            extensions = [line.strip() for line in file if line.startswith("- .")]
        return [ext[2:] for ext in extensions]


# --- Response Generation with Gemini ---
def truncate_context(text: str, max_length: int) -> str:
    """Truncates text to a maximum length while preserving complete sentences."""
    if len(text) <= max_length:
        return text

    truncated_text = text[:max_length]
    last_period_index = truncated_text.rfind(".")

    if last_period_index != -1:
        return truncated_text[: last_period_index + 1].strip()
    else:
        return truncated_text.strip()


def generate_response(query: str, context: str) -> Optional[str]:
    """Generates a response using a Google Gemini model and provided context."""
    try:
        truncated_context = truncate_context(context, MAX_CONTEXT_LENGTH)

        configure(api_key=os.environ["GEMINI_API_KEY"])
        generation_config = GenerationConfig(**RAGConfig.GENERATION_CONFIG)
        model = GenerativeModel(
            model_name=RAGConfig.MODEL_NAME, generation_config=generation_config
        )
        chat = model.start_chat()
        response = chat.send_message(
            f"Context: {truncated_context}\n\nQuestion: {query}\n\nAnswer:"
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None


# --- FastAPI Application ---

app = FastAPI()


class AppState:
    limiter: Limiter = None


app.state = AppState()
app.state.limiter = Limiter(key_func=get_remote_address)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(data: dict, expires_delta: Optional[int] = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def get_current_user(
        token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    """Decodes the JWT token and retrieves the current user."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        print(f"Decoded Payload: {payload}")
        username_payload = payload.get("sub")
        if username_payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    user = db.query(User).filter(User.username == username_payload).first()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return user


async def authenticate_user(
        username: str, password: str, db: Session
) -> Optional[User]:
    """Authenticates a user against the database."""
    # Correctly query the database for the user
    user = db.query(User).filter(User.username.__eq__(username)).first()

    # Check if user exists and verify the password
    if user and pwd_context.verify(password, user.hashed_password):
        return user
    else:
        return None


# --- Pydantic Models ---


class UserCreate(BaseModel):
    username: str
    password: str


class RepoFilesParams(BaseModel):
    repo_full_name: str = Query(
        ..., description="The full name of the repository (e.g., 'owner/repo')"
    )
    branch: Optional[str] = Query(
        "main", description="The branch to retrieve files from"
    )
    file_path: Optional[str] = Query(
        None, description="Optional path to a specific file or directory"
    )


class SearchDatabasePaginatedParams(BaseModel):
    query: str = Query(..., description="The search query")
    skip: int = Query(0, ge=0, description="Number of records to skip")
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of records to return"
    )


class FileInfo(BaseModel):
    filename: str
    content: str


class RepoFilesResponse(BaseModel):
    files: Dict[str, str]


# --- API Endpoints ---


@app.post(f"{API_V1_PREFIX}/token", tags=["Auth"])
async def login(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db),
):
    """Endpoint to obtain a bearer token for authentication."""
    user = await authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise InvalidCredentials()  # Raise custom exception
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post(f"{API_V1_PREFIX}/register", tags=["Auth"])
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Registers a new user."""

    hashed_password = pwd_context.hash(user_data.password)
    db_user = User(username=user_data.username, hashed_password=hashed_password)
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        # Generate the JWT token
        access_token = create_access_token(data={"sub": user_data.username})
        return {"message": "User registered successfully", "access_token": access_token}
    except IntegrityError as e:
        db.rollback()
        logging.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )


@app.post(f"{API_V1_PREFIX}/google_login", tags=["Auth"])
async def google_login(token_id: str = Form(...), db: Session = Depends(get_db)):
    """Authenticates a user using Google Sign-In."""
    try:
        id_info = id_token.verify_oauth2_token(
            token_id, requests.Request(), GOOGLE_CLIENT_ID
        )

        # Extract user info
        userid = id_info["sub"]
        email = id_info["email"]
        # You might want to verify other claims like 'aud' and 'iss' here

        # Find or create the user in your database
        user = db.query(User).filter(User.username == email).first()
        if not user:
            # Create a new user if this is their first login
            hashed_password = pwd_context.hash(userid)  # Test double password for now
            new_user = User(username=email, hashed_password=hashed_password)
            db.add(new_user)
            db.commit()
            user = new_user

        # Create JWT token
        access_token = create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}

    except ValueError as e:
        # Invalid token
        logging.error(f"Google login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


@app.post(
    f"{API_V1_PREFIX}/generate/",
    tags=["RAG"],
    responses={
        200: {"description": "Answer generated successfully"},
        500: {"description": "An error occurred during answer generation"},
    },
)
@app.state.limiter.limit("5/minute", error_message="Rate limit exceeded")
async def generate_answer(
        request: Request,
        repo_full_name: str = Query(...),
        query: str = Query(...),
        branch: Optional[str] = Query("main"),
        file_path: Optional[str] = Query(None),
        # current_user=Depends(get_current_user),  # Apply authentication
        db: Session = Depends(get_db),  # Inject database session
        github_repo: GitHubRepository = Depends(
            get_github_repository
        ),  # Inject GitHub repo
        rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """Generates an answer to a query using code from a GitHub repository."""
    spinner = Halo(text="Processing...", spinner="dots")
    try:
        spinner.start()

        # Fetch files based on a path (if provided) or entire repo
        if file_path:
            # Fetch content from the database
            file = db.query(File).filter_by(path=file_path, branch_id=branch).first()
            if file:
                context = {file.path: file.content}
            else:
                # If not in the database, fetch from GitHub
                context = await github_repo.get_files_in_path(
                    branch=branch, path=file_path
                )
                # Store the file content in the database
                branch_obj = db.query(Branch).filter_by(name=branch).first()
                if branch_obj is None:
                    branch_obj = Branch(name=branch)
                    db.add(branch_obj)
                    db.commit()
                    db.refresh(branch_obj)
                new_file = File(
                    name=os.path.basename(file_path),
                    path=file_path,
                    content=context[file_path],
                    branch_id=branch_obj.id,
                )
                db.add(new_file)
                db.commit()
        else:
            # Check if the entire repo is in the database
            repository = (
                db.query(Repository).filter_by(full_name=repo_full_name).first()
            )
            if repository:
                # If the repo exists, get files from the database
                files = (
                    db.query(File)
                    .filter_by(
                        branch_id=repository.branches[0].id
                    )  # Use repository.branches[0].id
                    .all()
                )
                context = {f.path: f.content for f in files}
            else:
                # Download and extract the repo if not in the database
                context = await github_repo.download_and_extract_zip(branch=branch)
                # Store the repository, branch, and files in the database
                new_repository = Repository(full_name=repo_full_name)
                db.add(new_repository)
                db.commit()
                db.refresh(new_repository)

                branch_obj = Branch(name=branch, repository_id=new_repository.id)
                db.add(branch_obj)
                db.commit()
                db.refresh(branch_obj)

                for file_path, file_content in context.items():
                    new_file = File(
                        name=os.path.basename(file_path),
                        path=file_path,
                        content=file_content,
                        branch_id=branch_obj.id,
                    )
                    db.add(new_file)
                    db.commit()

        # --- Code Chunking and Embeddings ---
        embedder = SentenceTransformer("all-mpnet-base-v2")
        query_embedding = embedder.encode(query)
        code_chunks = []
        code_embeddings = []

        for filename, code in context.items():
            if not isinstance(code, str):
                logging.warning(
                    f"Code content in {filename} is not a string, skipping."
                )
                continue

            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        start_line = node.lineno
                        end_line = getattr(node, "end_lineno", start_line)
                        chunk = code.splitlines()[start_line - 1: end_line]
                        code_chunks.append(" ".join(chunk))
            except SyntaxError:
                logging.warning(f"SyntaxError while parsing {filename}")

        # Check if code_chunks is empty
        if not code_chunks:
            logging.error(
                "No code chunks found. Ensure the repository contains valid code."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No code chunks found in the repository.",
            )

        for chunk in code_chunks:
            chunk_embedding = embedder.encode(chunk)
            code_embeddings.append(chunk_embedding)

        # Check if code_embeddings is empty
        if not code_embeddings:
            logging.error(
                "No code embeddings found. Ensure the repository contains valid code."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No code embeddings found in the repository.",
            )

        code_embeddings = np.array(code_embeddings)  # Combine into a single array
        query_embedding = torch.tensor(query_embedding)  # Convert to tensor
        code_embeddings = torch.tensor(code_embeddings)  # Convert to tensor

        # Log the shapes of the embeddings
        logging.info(f"Query embedding shape: {query_embedding.shape}")
        logging.info(f"Code embeddings shape: {code_embeddings.shape}")

        # Similarity Search
        similarities = util.cos_sim(query_embedding, code_embeddings)
        top_k = 5  # Number of top similar chunks to select
        top_k_indices = similarities[0].argsort(descending=True)[:top_k]

        # Concatenate the most relevant code chunks
        relevant_code_chunks = [code_chunks[i] for i in top_k_indices]
        relevant_code = "\n\n".join(relevant_code_chunks)

        # --- Truncate the context if needed ---
        truncated_context = truncate_context(relevant_code, MAX_CONTEXT_LENGTH)

        # --- Generate the response ---
        answer = generate_response(query, truncated_context)

        if answer:
            return {"answer": answer}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred.",
        )
    finally:
        spinner.stop()


@app.post(f"{API_V1_PREFIX}/search_database/", tags=["RAG"])
@app.state.limiter.limit("5/minute")
async def search_database(
        request: Request, query: str = Form(...), db: Session = Depends(get_db)
):
    """Searches the database for content matching a given query."""

    try:
        results = db.query(File).filter(File.content.contains(query)).all()
        if results:
            matched_contents = [result.content for result in results]
            return {"results": matched_contents}
        else:
            return {"results": [], "message": "No matching content found."}
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred.",
        )


@app.get(f"{API_V1_PREFIX}/repo_metadata/", tags=["GitHub"])
async def get_repo_metadata(
        repo_full_name: str = Query(
            ..., description="Full name of the repository (e.g., 'owner/repo')"
        ),
        github_repo: GitHubRepository = Depends(get_github_repository),
):
    """Fetches and returns metadata for a given GitHub repository."""
    try:
        metadata = await github_repo.get_repo_metadata()
        if metadata:
            return metadata
        else:
            raise RepositoryNotFound(repo_full_name)
    except GitHubAPIError as e:
        raise e


@app.get(
    f"{API_V1_PREFIX}/repo_files/",
    tags=["GitHub"],
    response_model=RepoFilesResponse,
    responses={
        200: {"description": "Files retrieved successfully"},
        404: {"description": "Repository not found"},
        500: {"description": "An error occurred retrieving files"},
    },
)
async def list_repo_files(
        repo_files_params: RepoFilesParams = Depends(),
        github_repo: GitHubRepository = Depends(get_github_repository),
) -> RepoFilesResponse:
    """Lists the files in a given GitHub repository branch."""
    try:
        if repo_files_params.file_path:
            file_contents = await github_repo.get_files_in_path(
                branch=repo_files_params.branch, path=repo_files_params.file_path
            )
        else:
            file_contents = await github_repo.download_and_extract_zip(
                branch=repo_files_params.branch
            )
        return RepoFilesResponse(files=file_contents)
    except RepositoryNotFound as e:
        raise e
    except GitHubAPIError as e:
        raise e
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred.",
        )


@app.post(
    f"{API_V1_PREFIX}/search_database_paginated/",
    tags=["RAG"],
    responses={
        200: {"description": "Search results retrieved successfully"},
        500: {"description": "An error occurred during the search"},
        429: {"description": "Rate limit exceeded"},
    },
)
@app.state.limiter.limit("5/minute")
async def search_database_paginated(
        request: Request,
        search_params: SearchDatabasePaginatedParams = Depends(),
        db: Session = Depends(get_db),
        rate_limiter: RateLimiter = Depends(get_rate_limiter),
) -> dict:
    """Searches the database with pagination support."""
    try:
        rate_limiter.limit("5/minute")  # Apply rate limiting

        results = (
            db.query(File)
            .filter(File.content.contains(search_params.query))
            .offset(search_params.skip)
            .limit(search_params.limit)
            .all()
        )

        if results:
            matched_contents = [result.content for result in results]
            total_results = (
                db.query(File)
                .filter(File.content.contains(search_params.query))
                .count()
            )
            return {
                "results": matched_contents,
                "count": len(matched_contents),
                "total_results": total_results,
                "skip": search_params.skip,
                "limit": search_params.limit,
            }
        else:
            return {
                "results": [],
                "count": 0,
                "total_results": 0,
                "skip": search_params.skip,
                "limit": search_params.limit,
            }
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred.",
        )


@app.get(f"{API_V1_PREFIX}/rate_limit_status/", tags=["Rate Limiting"])
async def rate_limit_status(request: Request):
    """Provides information on the current rate limit status."""
    return {"rate_limit": "Rate limiting is enabled."}


# --- Error Handling ---
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handles Pydantic validation errors."""
    logging.error(f"Validation error: {exc}")
    return JSONResponse(
        content={"detail": exc.errors()},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handles rate limit exceeded errors."""
    logging.warning(f"Rate limit exceeded for {get_remote_address(request)}: {exc}")
    return JSONResponse(
        content={"detail": "Rate limit exceeded. Please wait and try again."},
        status_code=429,
    )


@app.exception_handler(GitHubAPIError)
async def github_api_error_handler(request: Request, exc: GitHubAPIError):
    """Handles GitHub API errors."""
    logging.error(f"GitHub API Error: {exc.detail}")
    return JSONResponse(content={"detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handles internal server errors."""
    logging.exception(exc)
    return JSONResponse(
        content={"detail": "An internal server error occurred."},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Performs startup tasks."""
    logging.info("Application starting up...")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logging.info(f"Connected to database: {DATABASE_URL}")
    logging.info(f"Tables: {tables}")
    for route in app.routes:
        if isinstance(route, APIRoute):
            logging.info(f"API Route: {route.path} [{route.methods}]")
    logging.info("Application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """Performs cleanup tasks during shutdown."""
    logging.info("Application shutting down...")
    logging.info("Application shutdown complete.")


# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
