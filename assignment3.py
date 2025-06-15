import openai
import json
from typing import List
from pydantic import BaseModel, ValidationError
import secrets  # Import your Azure secrets

# Azure OpenAI client initialization
client = openai.AzureOpenAI(
    api_key=secrets.AZURE_OPENAI_API_KEY,
    api_version=secrets.AZURE_OPENAI_API_VERSION,
    azure_endpoint=secrets.AZURE_OPENAI_ENDPOINT,
)

# Pydantic model for schema validation
class MeetingNotes(BaseModel):
    summary: str
    action_items: List[str]

# Much more deterministic prompt
def build_prompt(transcript: str) -> str:
    return f"""
    Given the following meeting transcript:

    {transcript}

    Generate a JSON object with exactly two keys:
    - "summary": A two-sentence summary of the discussion.
    - "action_items": A list of action items, each starting with a dash.

    Output only valid JSON.
    """

# Extraction function — fully indented and correct
def extract_meeting_notes(transcript: str) -> dict:
    prompt = build_prompt(transcript)

    response = client.chat.completions.create(
        model=secrets.AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}  # Azure requires this format
    )

    parsed_json = response.choices[0].message.content
    parsed_dict = json.loads(parsed_json)
    validated = MeetingNotes(**parsed_dict)
    return parsed_dict

# Sample test run — fully indented
if __name__ == "__main__":
    sample_transcript = """
    Alice: Welcome everyone. Today we need to finalize the Q3 roadmap.
    Bob: I’ve emailed the updated feature list—please review by Friday.
    Carol: I’ll set up the user‐testing sessions next week.
    Dan: Let’s push the new UI mockups to staging on Wednesday.
    Alice: Great. Also, can someone compile the stakeholder feedback into a slide deck?
    Bob: I can handle the slide deck by Monday.
    Alice: Thanks, team. Meeting adjourned.
    """

    result = extract_meeting_notes(sample_transcript)
    print(json.dumps(result, indent=2))



