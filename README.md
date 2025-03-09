# HTML to Markdown Conversion with AWS Bedrock

This project provides a solution for converting HTML content into Markdown format using AWS Bedrock's Titan model. It handles large HTML documents by splitting them into manageable chunks, preserves metadata, and ensures that token limits are respected.

## Overview

The script performs the following tasks:

- **Escape Sequence Correction:**  
  Automatically fixes invalid escape sequences in the input HTML.

- **Metadata Extraction:**  
  Extracts metadata (e.g., title and meta tags) from the HTML content using BeautifulSoup.

- **HTML Preprocessing:**  
  Strips the `<head>` section to focus on the content of the `<body>`, avoiding duplicate metadata.

- **Chunk-Based Conversion:**  
  Splits the HTML content into chunks if it exceeds a defined token limit, using a sliding window approach with token overlaps for context.

- **Conversion with AWS Bedrock:**  
  Converts HTML chunks into Markdown format using the AWS Bedrock Titan model, following specific Markdown conversion rules.

## Features

- **Automatic Fixes:**  
  Corrects common escape sequence issues in HTML.

- **Metadata Preservation:**  
  Extracts and incorporates metadata as YAML front matter in the Markdown output.

- **Chunked Processing:**  
  Efficiently handles large HTML documents by processing them in chunks while maintaining context between chunks.

- **Customizable Token Limits:**  
  Uses the `tiktoken` library to enforce token limits and split the HTML accordingly.

## Prerequisites

- **Python Version:** 3.7 or higher
- **AWS Credentials:**  
  Ensure that your AWS credentials are properly configured (e.g., via AWS CLI or environment variables).
- **Libraries:**  
  - `boto3`
  - `tiktoken`
  - `beautifulsoup4`
  - `re`
  - `json`
  - `warnings`

Install the required packages using pip:

```bash
pip install boto3 tiktoken beautifulsoup4
