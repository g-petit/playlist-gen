from autoevals import LLMClassifier
from braintrust import Eval, init_dataset, Score
from main import run_agent, DEFAULT_SYSTEM_PROMPT, DEFAULT_MODEL

def task(input: dict, hooks):
    user_input = input.get("user_request")
    parameters = hooks.parameters if hooks else {}

    def get_param(key, default):
        val = (parameters or {}).get(key)
        # Playground passes plain strings; standard eval injects raw schema dicts —
        # only use the value if it's actually a string
        return val if isinstance(val, str) and val else default

    system_prompt = get_param("system_prompt", DEFAULT_SYSTEM_PROMPT)
    llm_model = get_param("llm_model", DEFAULT_MODEL)
    return run_agent(user_input, system_prompt, llm_model)

variety_scorer = LLMClassifier(
    name="Variety",
    prompt_template="""You are evaluating a music playlist for artist and genre diversity.

Here is the playlist:
{{output}}

Rate the variety of this playlist:
- Choose "High" if the playlist includes a good mix of different artists and genres with minimal repetition.
- Choose "Low" if the playlist is repetitive, dominated by one or two artists, or lacks genre diversity.""",
    choice_scores={"High": 1, "Low": 0},
    use_cot=True,
)

def playlist_length_scorer(output: dict):
    playlist = output.get("playlist")
    if not playlist:
        return Score(name="PlaylistLength", score=None)
    duration = playlist.get("total_duration_min", 0)
    score = 1 if duration <= 30 else 0
    return Score(name="PlaylistLength", score=score)

Eval(
    name="PlaylistGenerator",
    task=task,
    data=init_dataset(project="PlaylistGenerator", name="InputExamples"),
    scores=[variety_scorer, playlist_length_scorer],
    parameters={
        "system_prompt": {
            "type": "string",
            "description": "System prompt for the playlist agent",
            "default": DEFAULT_SYSTEM_PROMPT,
        },
        "llm_model": {
            "type": "string",
            "description": "Model to use (e.g. claude-haiku-4-5, gpt-4o-mini)",
            "default": DEFAULT_MODEL,
        },
    },
)
