import warnings
import boto3
import json
import tiktoken  # OpenAI tokenizer, adjust if using a different tokenizer
from bs4 import BeautifulSoup  # For metadata extraction and HTML processing
import re

# Suppress SyntaxWarnings (specifically invalid escape sequence warning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Initialize AWS Bedrock client using default AWS credentials
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"  # Adjust region as needed
)

# Initialize tokenizer (ensure compatibility with your LLM)
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN_LIMIT = 4096  # Example token limit for the model


# Function to automatically remove invalid escape sequences
def fix_escape_sequences(input_string):
    # Replace all invalid escape sequences like \:
    fixed_string = re.sub(r'\\:', ':', input_string)  # Fix specifically for '\:'
    # Remove unwanted escape sequences like \n, \t, etc.
    fixed_string = re.sub(r'\\n', '', fixed_string)  # Remove '\n' escape
    fixed_string = re.sub(r'\\t', '', fixed_string)  # Remove '\t' escape
    fixed_string = re.sub(r'\\r', '', fixed_string)  # Remove '\r' escape (if needed)

    return fixed_string


# Extract metadata from HTML content
def extract_metadata(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    metadata = {}
    if soup.title and soup.title.string:
        metadata["title"] = soup.title.string.strip()
    for meta in soup.find_all("meta"):
        if meta.get("name"):
            metadata[meta["name"].strip()] = meta.get("content", "").strip()
    return metadata


def strip_head(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    if soup.body:
        return str(soup.body)
    return html_content


def is_html(content):
    soup = BeautifulSoup(content, "html.parser")
    return bool(soup.find())


def convert_chunk_to_markdown(html_chunk, previous_markdown="", metadata_block="", include_metadata=False):
    """
    Converts an HTML chunk to Markdown using AWS Bedrock.
    For the first chunk, includes the YAML front matter (metadata_block) at the top.
    The prompt instructs the model to:
      - Use explicit Markdown syntax for headings: h1 as "# ", h2 as "## ", etc.
      - Convert paragraphs to plain text.
      - Convert unordered lists to Markdown lists with "- " prefixes.
      - Output only one YAML block at the very beginning (if include_metadata is True).
      - Omit any duplicate metadata or YAML blocks from the body.
      - Not include the special token "END" in the final output.
    """
    if previous_markdown:
        prompt = f"""
You are a conversion engine that continues converting HTML to Markdown from previous output.
Rules:
- Do NOT output any YAML front matter in this chunk.
- Convert the following HTML to Markdown:
    - For <h1>, use "# ".
    - For <h2>, use "## ".
    - For <h3>, use "### ".
    - For paragraphs (<p>), use plain text.
    - For unordered lists (<ul> and <li>), use Markdown lists with "- ".
    - DO NOT include any raw HTML tags.
- Do NOT output the special token "END".
Previous Markdown:
{previous_markdown}

Now convert the following HTML to Markdown:
{html_chunk}

Markdown Output:
"""
    else:
        prompt = f"""
You are a conversion engine that converts HTML to Markdown.
Rules:
- Start the output with a YAML front matter block containing the following metadata exactly as provided:
{metadata_block if include_metadata else ""}

  * The YAML block must begin with a line with only '---', followed by key: "value" pairs (one per line), then a line with only '---', then an empty line.
- Convert the following HTML to Markdown:
    - For <h1>, use "# ".
    - For <h2>, use "## ".
    - For <h3>, use "### ".
    - For paragraphs (<p>), use plain text.
    - For unordered lists (<ul> and <li>), use Markdown lists with "- ".
    - Do NOT include any raw HTML tags.
    - Do NOT output the special token "END".
Convert the following HTML to Markdown:
{html_chunk}

Markdown Output:
"""
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 1000,  # Limit the input to 1000 tokens per chunk
            "temperature": 0.3,  # More deterministic output
            "topP": 0.9
        }
    })

    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-tg1-large",  # Use your desired model ID
            contentType="application/json",
            accept="application/json",
            body=body
        )
        response_body = response['body'].read().decode('utf-8')
        response_json = json.loads(response_body)
        markdown_output = response_json["results"][0]["outputText"].strip()
    except Exception as e:
        print(f"Error during model invocation: {e}")
        return ""
    return markdown_output


def process_html_in_chunks(html_content, max_chunk_size=1000, overlap_tokens=100, with_metadata=True):
    """
    Processes the full HTML content in chunks using a sliding window strategy.
    """
    # Handle empty content case: Specifically check if <body> tag is empty
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if not body_content or not body_content.get_text(strip=True):  # If body is empty or contains no meaningful text
        print("Error: No content to process.")
        return "", {}  # Return empty output if the HTML content is empty

    # Token count check to avoid exceeding the token limit
    total_tokens = len(tokenizer.encode(html_content))
    print(f"Debug: Total tokens = {total_tokens}")  # Debugging token count

    if total_tokens > MAX_TOKEN_LIMIT:
        print(f"Token limit exceeded ({total_tokens} tokens), splitting content into chunks.")

    # Check if the content exceeds the token limit and handle chunking
    if total_tokens > MAX_TOKEN_LIMIT:
        # Extract metadata from the full HTML
        metadata = extract_metadata(html_content)
        metadata_block = ""
        if with_metadata and metadata:
            metadata_lines = [f'{key}: "{value}"' for key, value in metadata.items()]
            metadata_block = "---\n" + "\n".join(metadata_lines) + "\n---\n\n"  # YAML front matter block

        # Remove <head> so that metadata is not in the body conversion
        body_html = strip_head(html_content)

        # Split content into chunks using the sliding window strategy
        markdown_chunks = []
        tokens = tokenizer.encode(body_html)
        total_tokens = len(tokens)
        start = 0
        previous_markdown = ""
        chunk_index = 0

        while start < total_tokens:
            end = min(start + max_chunk_size, total_tokens)
            # Use an overlap for context (except for the first chunk)
            if start > 0:
                chunk_tokens = tokens[start - overlap_tokens:end]
            else:
                chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            include_metadata = (chunk_index == 0)
            markdown_chunk = convert_chunk_to_markdown(
                html_chunk=chunk_text,
                previous_markdown=previous_markdown,
                metadata_block=metadata_block,
                include_metadata=include_metadata
            )
            if markdown_chunk.strip() == "":
                break  # Stop processing if no content is generated
            markdown_chunks.append(markdown_chunk)
            previous_markdown = markdown_chunk  # update context
            if "END" in markdown_chunk:
                break
            start += max_chunk_size
            chunk_index += 1

        aggregated_markdown = "\n".join(markdown_chunks)
        # Post-process: Remove any occurrence of the "END" token if present
        aggregated_markdown = aggregated_markdown.replace("END", "").strip()

        # Ensure the aggregated markdown begins with the YAML front matter.
        if with_metadata and metadata and not aggregated_markdown.startswith('---'):
            aggregated_markdown = metadata_block + aggregated_markdown

        return aggregated_markdown, metadata
    else:
        print("Token limit not exceeded. Processing as a single chunk.")
        markdown_output, metadata = convert_chunk_to_markdown(html_content, with_metadata=True)
        return markdown_output, metadata


if __name__ == "__main__":
    # Simulate a large document that exceeds the token limit
    html_content = """

"""

    try:
        aggregated_markdown, metadata = process_html_in_chunks(
            html_content, max_chunk_size=1000, overlap_tokens=100, with_metadata=True
        )

        # Print the aggregated markdown content and extracted metadata
        print("----- Aggregated Markdown Output -----")
        print(aggregated_markdown)
        print("\n----- Extracted Metadata -----")
        print(metadata)

    except ValueError as e:
        # Handle the error when the token limit is exceeded
        print(f"Error: {e}")
