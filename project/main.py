import io
import sys
import json
import zipfile
import requests
import frontmatter

from chunking import (
    simple_character_chunking,
    paragraph_sliding_window_chunking,
    section_chunking,
)


def read_repo_data(repo_owner, repo_name):
    """
    Download and parse all markdown files from a GitHub repository.
    
    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name
    
    Returns:
        List of dictionaries containing file content and metadata
    """
    prefix = 'https://codeload.github.com' 
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)
    
    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    
    for file_info in zf.infolist():
        filename = file_info.filename
        filename_lower = filename.lower()

        if not (filename_lower.endswith('.md') 
            or filename_lower.endswith('.mdx')):
            continue
    
        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode('utf-8', errors='ignore')
                post = frontmatter.loads(content)
                data = post.to_dict()
                data['filename'] = filename
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    zf.close()
    return repository_data





if __name__ == "__main__":

    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]

    data = read_repo_data(repo_owner, repo_name)
    base_name = f"{repo_owner}_{repo_name}"

    output_file = f"{base_name}_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    simple_chunks = simple_character_chunking(data, size=2000, step=1000)
    paragraph_chunks = paragraph_sliding_window_chunking(data, window_size=5, step=3)
    section_chunks = section_chunking(data, level=2)

    with open(f"{base_name}_simple_chunks.json", "w", encoding="utf-8") as f:
        json.dump(simple_chunks, f, indent=2, ensure_ascii=False)

    with open(f"{base_name}_paragraph_chunks.json", "w", encoding="utf-8") as f:
        json.dump(paragraph_chunks, f, indent=2, ensure_ascii=False)

    with open(f"{base_name}_section_chunks.json", "w", encoding="utf-8") as f:
        json.dump(section_chunks, f, indent=2, ensure_ascii=False)

    print(f"Documents: {len(data)}")
    print(f"Simple chunks: {len(simple_chunks)}")
    print(f"Paragraph chunks: {len(paragraph_chunks)}")
    print(f"Section chunks: {len(section_chunks)}")




