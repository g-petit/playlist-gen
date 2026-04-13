from autoevals import LLMClassifier
from braintrust import Eval, init_dataset, Score
from pydantic import BaseModel, Field
from main import run_agent, DEFAULT_SYSTEM_PROMPT, DEFAULT_MODEL

class SystemPromptParam(BaseModel):
    """Exposes the system prompt as an editable field in the Braintrust Playground."""
    value: str = Field(default=DEFAULT_SYSTEM_PROMPT, description="System prompt for the playlist agent")

class ModelParam(BaseModel):
    """Exposes the model as an editable field in the Braintrust Playground."""
    value: str = Field(default=DEFAULT_MODEL, description="Model to use (e.g. claude-haiku-4-5, gpt-4o-mini)")

def task(input: dict, hooks):
    user_input = input.get("user_request")
    parameters = hooks.parameters if hooks else {}
    system_prompt = parameters.get("system_prompt") if parameters else None
    llm_model = parameters.get("llm_model") if parameters else None
    return run_agent(
        user_input,
        system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT,
        llm_model if llm_model else DEFAULT_MODEL,
    )

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
        "system_prompt": SystemPromptParam,
        "llm_model": ModelParam,
    },
)
