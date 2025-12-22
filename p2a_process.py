import os
import re
import subprocess
import datetime
import glob
from anthropic import Anthropic
from openai import OpenAI

import pymupdf

claude = Anthropic()
openai_client = OpenAI()

GITHUB_PAGES_URL = "https://raw.githubusercontent.com/oaks3000/p2a-feed/main"

def extract_text_from_pdf(filepath):
    doc = pymupdf.open(filepath)
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text = full_text + text + "\n"
    return full_text

def clean_text(text):
    lines = text.split("\n")
    all_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0:
            continue
        if stripped.isdigit():
            continue
        line_lower = stripped.lower()
        if "downloaded from" in line_lower:
            continue
        if "jstor.org" in line_lower:
            continue
        if "all use subject to" in line_lower:
            continue
        if "linked references are available" in line_lower:
            continue
        if " x " in stripped and "=" in stripped:
            continue
        letter_count = sum(1 for char in stripped if char.isalpha())
        if len(stripped) > 0:
            letter_ratio = letter_count / len(stripped)
            if letter_ratio < 0.5:
                continue
        all_lines.append(stripped)
    references_index = len(all_lines)
    for i, line in enumerate(all_lines):
        line_lower = line.lower().strip()
        if line_lower in ["references", "bibliography"]:
            if i > len(all_lines) * 0.2:
                references_index = i
                break
    all_lines = all_lines[:references_index]
    cleaned_lines = []
    for line in all_lines:
        if len(line) < 20 and len(cleaned_lines) > 0:
            cleaned_lines[-1] = cleaned_lines[-1] + " " + line
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def find_sections(text):
    sections = {}
    roman_order = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    potential_headings = []
    for match in re.finditer(r'(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+([A-Z][A-Za-z\s\-\?]+)', text):
        numeral = match.group(1)
        title = match.group(2).strip()
        start_pos = match.start()
        potential_headings.append({
            "numeral": numeral,
            "title": title,
            "position": start_pos
        })
    headings = []
    expected_index = 0
    for h in potential_headings:
        numeral = h["numeral"]
        if numeral in roman_order:
            numeral_index = roman_order.index(numeral)
            if numeral_index == expected_index:
                headings.append(h)
                expected_index = expected_index + 1
    if len(headings) >= 3:
        sections["preamble"] = text[:headings[0]["position"]].strip()
        for i, heading in enumerate(headings):
            section_name = f"{heading['numeral']}. {heading['title']}"
            if i + 1 < len(headings):
                end_pos = headings[i + 1]["position"]
            else:
                end_pos = len(text)
            content_start = heading["position"] + len(heading["numeral"]) + 2 + len(heading["title"])
            section_content = text[content_start:end_pos].strip()
            sections[section_name.lower()] = section_content
        return sections
    lines = text.split("\n")
    heading_positions = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 10 or len(stripped) > 60:
            continue
        words = stripped.split()
        if len(words) < 2:
            continue
        capitalized_words = sum(1 for w in words if w[0].isupper())
        ratio = capitalized_words / len(words)
        if ratio >= 0.7 and not stripped.endswith(".") and not stripped.endswith(","):
            pos = text.find(stripped)
            if pos != -1:
                heading_positions.append({
                    "title": stripped,
                    "position": pos
                })
    if len(heading_positions) >= 3:
        sections["preamble"] = text[:heading_positions[0]["position"]].strip()
        for i, heading in enumerate(heading_positions):
            if i + 1 < len(heading_positions):
                end_pos = heading_positions[i + 1]["position"]
            else:
                end_pos = len(text)
            content_start = heading["position"] + len(heading["title"])
            section_content = text[content_start:end_pos].strip()
            if len(section_content) > 100:
                sections[heading["title"].lower()] = section_content
        return sections
    words = text.split()
    chunk_size = 3000
    if len(words) <= chunk_size:
        sections["full paper"] = text
    else:
        num_chunks = (len(words) // chunk_size) + 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            sections[f"part {i + 1}"] = chunk_text
    return sections

def summarise_section(section_name, section_text):
    if section_name.lower() in ["preamble", "title", "title page"]:
        return None
    section_lower = section_name.lower()
    if any(word in section_lower for word in ["method", "data", "design", "sample"]):
        section_guidance = "Keep this brief—one short paragraph. Focus only on what's essential to understand the findings. Skip routine procedural details."
    elif any(word in section_lower for word in ["result", "finding", "analysis"]):
        section_guidance = "Keep this to one or two paragraphs. Highlight the key findings in plain language. Only expand if something is genuinely surprising or important."
    elif any(word in section_lower for word in ["discussion", "conclusion", "implication"]):
        section_guidance = "This is important—give it proper attention. What are the takeaways? What should the listener remember?"
    else:
        section_guidance = "Summarise the key points concisely."
    prompt = f"""Summarise this section of an academic paper for a podcast-style audio format.

SECTION: "{section_name}"

STYLE:
- Sound like a knowledgeable friend explaining over coffee, not a lecturer
- Casual but authoritative—you know this stuff well
- Short, punchy sentences that work when spoken aloud
- No jargon without a quick plain-English explanation

RULES:
- Jump straight into the content. Never say things like "This section covers..." or "I'll summarise..."
- Never reference figures, tables, or equation numbers
- Convert statistics to plain language ("about twice as likely" not "OR = 2.1, p < 0.05")
- Add brief critical observations where relevant ("One limitation here is..." or "What's clever about this approach is...")
- Never use phrases like "here's the thing", "here's the kicker", "here's what's interesting", "the bottom line"
- {section_guidance}

TEXT:
{section_text}

Give your summary now, starting directly with the substance:"""
    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

def review_and_tighten(full_script):
    prompt = f"""Review this podcast script and remove repetition. 

The script was generated section-by-section, so it likely repeats:
- The study's purpose or research question
- Methodology descriptions
- Key findings mentioned multiple times

Edit the script to:
1. Keep the first mention of any repeated concept, remove or heavily condense subsequent mentions
2. Remove phrases like "here's the thing", "here's the kicker", "here's what's interesting", "the bottom line"
3. Smooth transitions between sections
4. Keep the casual, conversational tone
5. Aim for 5-10 minutes of audio (roughly 700-1400 words)

One or two light callbacks to earlier points is fine, but no redundant explanations.

Return ONLY the edited script, nothing else.

SCRIPT:
{full_script}"""
    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

def text_to_speech(text, output_filename, voice="nova"):
    max_chars = 4000
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 < max_chars:
            current_chunk = current_chunk + para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    print(f"  Generating audio in {len(chunks)} chunks...")
    temp_files = []
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i + 1}/{len(chunks)}...")
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=chunk
        )
        temp_filename = f"temp_chunk_{i}.mp3"
        with open(temp_filename, "wb") as f:
            f.write(response.content)
        temp_files.append(temp_filename)
    print("  Combining audio chunks...")
    with open("temp_filelist.txt", "w") as f:
        for temp_file in temp_files:
            f.write(f"file '{temp_file}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", "temp_filelist.txt",
        "-c", "copy", output_filename
    ], capture_output=True)
    for temp_file in temp_files:
        os.remove(temp_file)
    os.remove("temp_filelist.txt")

def add_to_podcast_feed(title, audio_filename, description):
    feed_path = "feed.xml"
    with open(feed_path, "r") as f:
        feed_content = f.read()
    pub_date = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    audio_url = f"{GITHUB_PAGES_URL}/audio/{audio_filename}"
    description = description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    new_episode = f"""
    <item>
      <title>{title}</title>
      <description>{description}</description>
      <pubDate>{pub_date}</pubDate>
      <enclosure url="{audio_url}" type="audio/mpeg"/>
      <guid>{audio_url}</guid>
    </item>
    """
    feed_content = feed_content.replace(
        "</channel>",
        new_episode + "\n  </channel>"
    )
    with open(feed_path, "w") as f:
        f.write(feed_content)


# =============================================================================
# NEW: Obsidian Markdown Summary Generation
# =============================================================================

def extract_metadata(raw_text, cleaned_text):
    """Use Claude to extract paper metadata from the text."""
    # Take first ~3000 chars which typically contains title, authors, abstract
    header_text = raw_text[:3000]
    
    prompt = f"""Extract metadata from this academic paper header. Return ONLY a JSON object with these fields:
- title: the paper's title (clean it up if needed)
- authors: comma-separated list of author names
- year: publication year (4 digits, or "unknown")
- journal: journal/source name (or "unknown")
- doi: DOI if present (or "unknown")

If you can't find a field, use "unknown".

TEXT:
{header_text}

Return ONLY the JSON object, no explanation:"""

    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = message.content[0].text.strip()
    
    # Parse JSON response
    import json
    try:
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = re.sub(r'^```json?\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        metadata = json.loads(response_text)
    except json.JSONDecodeError:
        metadata = {
            "title": "unknown",
            "authors": "unknown", 
            "year": "unknown",
            "journal": "unknown",
            "doi": "unknown"
        }
    
    return metadata


def generate_obsidian_summary(cleaned_text, metadata, paper_name):
    """Generate a structured summary for the Obsidian template."""
    
    # Take a substantial portion of the paper for analysis
    analysis_text = cleaned_text[:12000] if len(cleaned_text) > 12000 else cleaned_text
    
    prompt = f"""Analyse this academic paper and provide structured information for a research summary template.

Return your response in this EXACT format with these section headers:

RESEARCH_QUESTION:
[1-3 sentences: What problem or question does this paper address?]

METHODOLOGY:
[1-2 sentences: How did the researchers approach the problem?]

KEY_FINDINGS:
[3 bullet points, each starting with "- ": the main results or conclusions]

CORE_ARGUMENT:
[2-3 sentences: What is the central thesis or contribution?]

EVIDENCE:
[1-2 sentences: What data or reasoning supports the claims?]

STRENGTHS:
[1-2 bullet points starting with "- "]

LIMITATIONS:
[1-2 bullet points starting with "- "]

OPEN_QUESTIONS:
[1-2 bullet points starting with "- ": What remains unresolved?]

SUMMARY:
[2-3 sentence overview in plain language]

PAPER TEXT:
{analysis_text}

Respond with ONLY the formatted sections above, nothing else:"""

    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response = message.content[0].text
    
    # Parse the response into sections
    sections = {}
    current_section = None
    current_content = []
    
    for line in response.split('\n'):
        line = line.strip()
        if line.endswith(':') and line[:-1].upper().replace('_', '') == line[:-1].replace('_', ''):
            # This looks like a section header
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[:-1]
            current_content = []
        elif current_section:
            current_content.append(line)
    
    # Don't forget the last section
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections


def create_obsidian_markdown(metadata, summary_sections, paper_name):
    """Create the Obsidian markdown file from metadata and summary."""
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = metadata.get("title", paper_name)
    
    # Clean title for use in filename
    safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
    
    # Extract sections with fallbacks
    research_question = summary_sections.get("RESEARCH_QUESTION", "")
    methodology = summary_sections.get("METHODOLOGY", "")
    key_findings = summary_sections.get("KEY_FINDINGS", "1. \n2. \n3. ")
    core_argument = summary_sections.get("CORE_ARGUMENT", "")
    evidence = summary_sections.get("EVIDENCE", "")
    strengths = summary_sections.get("STRENGTHS", "")
    limitations = summary_sections.get("LIMITATIONS", "")
    open_questions = summary_sections.get("OPEN_QUESTIONS", "")
    summary = summary_sections.get("SUMMARY", "")
    
    markdown_content = f"""---
title: "{title}"
authors: {metadata.get("authors", "")}
year: {metadata.get("year", "")}
journal: {metadata.get("journal", "")}
doi: {metadata.get("doi", "")}
date_read: {today}
status: unread
rating: 
tags:
  - paper
  - 
---

# {title}

## Citation
> [!info] Reference
> **Authors:** {metadata.get("authors", "")}
> **Year:** {metadata.get("year", "")}
> **Journal/Source:** {metadata.get("journal", "")}
> **DOI/Link:** {metadata.get("doi", "")}

---

## Research Question
*What problem or question does this paper address?*

{research_question}

## Methodology
*How did the researchers approach the problem?*

{methodology}

## Key Findings
*What are the main results or conclusions?*

{key_findings}

## Core Argument
*What is the central thesis or contribution?*

{core_argument}

## Evidence & Support
*What data or reasoning supports the claims?*

{evidence}

---

## Critical Notes

### Strengths

{strengths}

### Limitations

{limitations}

### Open Questions
*What remains unresolved or needs further investigation?*

{open_questions}

---

## Relevance & Connections

### Why This Matters
*How does this relate to your research or interests?*


### Related Papers
- [[]]
- [[]]

### Key Concepts
- [[]]

---

## Notable Quotes
> [!quote]
> "Quote here" (p. )

---

## Summary
*2-3 sentence overview in your own words*

{summary}
"""
    
    return markdown_content, safe_title


def process_paper(pdf_path):
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"Processing: {pdf_path}")
    print("")
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)
    print("Finding sections...")
    sections = find_sections(cleaned_text)
    print(f"Found {len(sections)} sections")
    print("")
    
    # ===========================================
    # NEW: Extract metadata and generate Obsidian summary
    # ===========================================
    print("Extracting paper metadata...")
    metadata = extract_metadata(raw_text, cleaned_text)
    print(f"  Title: {metadata.get('title', 'unknown')[:60]}...")
    
    print("Generating Obsidian summary...")
    summary_sections = generate_obsidian_summary(cleaned_text, metadata, paper_name)
    
    markdown_content, safe_title = create_obsidian_markdown(metadata, summary_sections, paper_name)
    
    # Save markdown file
    os.makedirs("summaries", exist_ok=True)
    markdown_filename = f"{paper_name}.md"
    markdown_path = f"summaries/{markdown_filename}"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"  Saved: {markdown_path}")
    print("")
    # ===========================================
    
    # Continue with existing audio generation
    summaries = {}
    for section_name, section_text in sections.items():
        print(f"Summarising: {section_name}...")
        summary = summarise_section(section_name, section_text)
        if summary is not None:
            summaries[section_name] = summary
    print("")
    clean_title = paper_name.replace("_", " ").replace("-", " ")
    intro_prompt = f"""Write a 2-sentence podcast intro for a paper called "{clean_title}". 
Be casual and intriguing—hook the listener. Don't say "welcome" or "today we're looking at". 
Just set up what the paper is about and why it's interesting. Start directly."""
    intro_message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[
            {"role": "user", "content": intro_prompt}
        ]
    )
    intro = intro_message.content[0].text
    full_script = f"{intro}\n\n"
    for section_name, summary in summaries.items():
        full_script = full_script + summary + "\n\n"
    print("Reviewing for repetition...")
    full_script = review_and_tighten(full_script)
    print("Converting to audio (this may take a moment)...")
    audio_filename = f"{paper_name}_summary.mp3"
    text_to_speech(full_script, audio_filename)
    os.makedirs("audio", exist_ok=True)
    os.rename(audio_filename, f"audio/{audio_filename}")
    print("Updating podcast feed...")
    description = full_script[:200].replace("\n", " ") + "..."
    add_to_podcast_feed(paper_name, audio_filename, description)
    os.makedirs("processed", exist_ok=True)
    os.rename(pdf_path, f"processed/{os.path.basename(pdf_path)}")
    print(f"Done processing: {paper_name}")
    print(f"  Audio: audio/{audio_filename}")
    print(f"  Summary: {markdown_path}")


pdf_files = glob.glob("papers/*.pdf")
if len(pdf_files) == 0:
    print("No new papers to process")
else:
    for pdf_path in pdf_files:
        process_paper(pdf_path)
