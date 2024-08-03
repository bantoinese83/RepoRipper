import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';



// --- RAG (Answer Generation) ---

export const generateAnswer = async (
    repoFullName: string,
    query: string,
    branch = 'main',
    filePath?: string,
) => {
    const response = await axios.post(`${API_BASE_URL}/generate/`, null, {
        params: { repo_full_name: repoFullName, query, branch, file_path: filePath },
    });
    return response.data;
};