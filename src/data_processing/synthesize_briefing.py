import os
import sys
import json
from datetime import datetime
import pytz
from dotenv import load_dotenv

# --- LLM API Client Imports ---
import google.generativeai as genai
import openai
import anthropic

# --- Add project root to path for local imports ---
# This allows us to import from the 'prompts' and 'config' directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- Configuration ---
# Load environment variables from the .env file in the project root
load_dotenv()

# Configure API clients
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    openai.api_key = os.getenv("OPENAI_API_KEY")
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except TypeError:
    print("Error: One or more API keys are not set in the .env file. Please check your configuration.")
    sys.exit(1)

# --- Define the models you want to query ---
# NOTE: The models you requested are not yet released.
# I am substituting them with the latest appropriate models available.
MODELS_TO_QUERY = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-4-5-20250929"
}

def get_file_paths():
    """Generates the absolute paths for today's data files."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    today_str = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    daily_folder_processed = os.path.join(project_root, 'data', 'daily_news', today_str, 'processed')
    daily_folder = os.path.join(project_root, 'data', 'daily_news', today_str, 'raw')
    os.makedirs(daily_folder_processed, exist_ok=True) # Ensure the directory exists

    return {
        "bias_input": os.path.join(daily_folder, 'daily_bias.json'),
        "news_input": os.path.join(daily_folder, 'news.json'),
        "prompt_template": os.path.join(project_root, 'prompts', 'prompts.json'),
        "processed_output": os.path.join(daily_folder_processed, 'processed_briefing.json')
    }

def construct_prompt(paths):
    """Loads raw data and constructs the final prompt for the LLMs."""
    try:
        with open(paths['bias_input'], 'r') as f:
            ml_bias_data = json.load(f)
        
        # NOTE: Using your updated file structure where news.json is in the daily folder
        with open(paths['news_input'], 'r') as f:
            news_data = json.load(f)
            # We only need the headlines for this prompt
            raw_headlines = [article['headline'] for article in news_data.get('macro', [])]
            headlines_text = "\n".join(f"- {h}" for h in raw_headlines)

        with open(paths['prompt_template'], 'r') as f:
            synthesis_prompt_template = json.load(f)['pre_market_synthesis_prompt']['content']

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure raw data files exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from a file: {e}")
        return None

    # Replace placeholders in the prompt template
    prompt = synthesis_prompt_template.replace(
        "{{ml_bias_data}}", json.dumps(ml_bias_data, indent=2)
    )
    prompt = prompt.replace("{{news_headlines}}", headlines_text)
    
    return prompt

def get_llm_opinions(prompt):
    """Queries the specified LLMs and returns their raw responses."""
    if not prompt:
        return None

    all_responses = {"generated_at_utc": datetime.utcnow().isoformat(), "model_responses": {}}
    
    print("\nQuerying LLMs for pre-market analysis...")

    # --- Gemini ---
    try:
        print(f"  - Requesting analysis from Gemini ({MODELS_TO_QUERY['gemini']})...")
        model = genai.GenerativeModel(MODELS_TO_QUERY['gemini'])
        response = model.generate_content(prompt)
        # Gemini often returns JSON wrapped in markdown, so we clean it
        cleaned_response = response.text.strip().replace('```json\n', '').replace('\n```', '')
        all_responses["model_responses"]["gemini"] = json.loads(cleaned_response)
        print("    ...Success.")
    except Exception as e:
        all_responses["model_responses"]["gemini"] = {"error": str(e)}
        print(f"    ...FAILED. Error: {e}")

    # --- OpenAI ---
    try:
        print(f"  - Requesting analysis from OpenAI ({MODELS_TO_QUERY['openai']})...")
        response = openai.chat.completions.create(
            model=MODELS_TO_QUERY['openai'],
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a Senior Trading Analyst providing structured JSON output."},
                {"role": "user", "content": prompt}
            ]
        )
        all_responses["model_responses"]["openai"] = json.loads(response.choices[0].message.content)
        print("    ...Success.")
    except Exception as e:
        all_responses["model_responses"]["openai"] = {"error": str(e)}
        print(f"    ...FAILED. Error: {e}")

    # # --- Anthropic ---
    # try:
    #     print(f"  - Requesting analysis from Anthropic ({MODELS_TO_QUERY['anthropic']})...")
    #     message = anthropic_client.messages.create(
    #         model=MODELS_TO_QUERY['anthropic'],
    #         max_tokens=4096,
    #         messages=[
    #              {"role": "user", "content": prompt}
    #         ]
    #     )
    #     # Anthropic doesn't have a forced JSON mode, so we extract it from the text block
    #     cleaned_response = message.content[0].text.strip().replace('```json\n', '').replace('\n```', '')
    #     all_responses["model_responses"]["anthropic"] = json.loads(cleaned_response)
    #     print("    ...Success.")
    # except Exception as e:
    #     all_responses["model_responses"]["anthropic"] = {"error": str(e)}
    #     print(f"    ...FAILED. Error: {e}")

    return all_responses

def main():
    """Main execution function."""
    print("="*50)
    print("Starting Pre-Market Synthesis Process")
    print("="*50)

    # 1. Get file paths
    paths = get_file_paths()
    print(f"Output will be saved to: {paths['processed_output']}")

    # 2. Construct the prompt
    prompt = construct_prompt(paths)
    if not prompt:
        sys.exit(1) # Exit if prompt construction failed

    # 3. Get opinions from all LLMs
    final_data = get_llm_opinions(prompt)

    # 4. Save the combined responses to a processed JSON file
    if final_data:
        try:
            with open(paths['processed_output'], 'w') as f:
                json.dump(final_data, f, indent=4)
            print(f"\nSuccessfully saved all LLM responses to {paths['processed_output']}")
        except Exception as e:
            print(f"Error saving the final JSON file: {e}")
    
    print("\nSynthesis process complete.")
    print("="*50)


if __name__ == "__main__":
    main()