import pymupdf4llm
import os
import re
import base64
import yaml
from mimetypes import guess_type
from openai import OpenAI

with open("config.yaml", "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

OPENAI_API_KEY = config.get("openai_key")

client = OpenAI(api_key=OPENAI_API_KEY)

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def generate_image_description(image_url, image_path, page_num):
    prompt = f"""Extract only the factual data and content visible in this image.
Do not add interpretations, insights, opinions, or analysis.
For charts or graphs: list the exact values, labels, axis names, and data points as shown.
For tables: reproduce the table data as-is in a structured format.
For mathematical formulas or equations: write the exact formula in LaTeX notation.
For architecture diagrams or flowcharts: list the components and their connections as shown.
If you cannot extract meaningful data, respond with exactly 'None'.
This image is from page {page_num} of the document."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """You extract factual data from academic document images.
Output only what is visible in the image. Do not interpret or summarize.""",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                ],
            },
        ],
        max_tokens=1500,
    )

    return response.choices[0].message.content.strip()


def replace_images_with_descriptions(text, page_num):
    """Replace markdown image references with [[IMAGE_DATA_START]]...[[IMAGE_DATA_END]] format."""
    def replace_match(match):
        img_path = match.group(1)
        if not os.path.exists(img_path):
            return match.group(0)

        image_url = local_image_to_data_url(img_path)
        description = generate_image_description(image_url, img_path, page_num)
        print(f"Page {page_num} | {os.path.basename(img_path)}")

        return f"[[IMAGE_DATA_START]]Image URL: {img_path}, Image description: {description}[[IMAGE_DATA_END]]"

    return re.sub(r'!\[.*?\]\((.*?)\)', replace_match, text)


def parse_pdf(pdf_path, text_output_file="extracted_text.txt", image_output_dir="extracted_images"):
    os.makedirs(image_output_dir, exist_ok=True)

    md_pages = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
        write_images=True,
        image_path=image_output_dir,
    )

    print(f"Parsing {pdf_path} ({len(md_pages)} pages)")

    with open(text_output_file, "w", encoding="utf-8") as f:
        for i, page in enumerate(md_pages):
            page_num = i + 1
            page_text = page.get("text", "")

            # Replace image markdown with [[IMAGE_DATA_START]] format & descriptions
            img_count = len(re.findall(r'!\[.*?\]\(.*?\)', page_text))
            if img_count > 0:
                page_text = replace_images_with_descriptions(page_text, page_num)

            f.write(f"\n\n---\n\n<!-- PAGE {page_num} -->\n\n")
            f.write(page_text)

    # image_files = [f for f in os.listdir(image_output_dir) if f.endswith(".png")]
    print("Completed... Extracted text and images")


if __name__ == "__main__":
    parse_pdf(
        pdf_path="02556v1.pdf",
        text_output_file="extracted_text.txt",
        image_output_dir="extracted_images",
    )
