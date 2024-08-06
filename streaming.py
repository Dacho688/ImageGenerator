from transformers.agents.agent_types import AgentAudio, AgentImage, AgentText, AgentType
from transformers.agents import CodeAgent
import spaces


@spaces.GPU
def stream_to_gradio(agent: CodeAgent, task: str, **kwargs):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""

    try:
        from gradio import ChatMessage
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    class Output:
        output: AgentType | str = None

    Output.output = agent.run(task,**kwargs)
    if isinstance(Output.output, AgentText):
        yield ChatMessage(role="assistant", content=f"{Output.output}")
    elif isinstance(Output.output, AgentImage):
        yield ChatMessage(
            role="assistant",
            content={"path": Output.output.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(Output.output, AgentAudio):
        yield ChatMessage(
            role="assistant",
            content={"path": Output.output.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield ChatMessage(role="assistant", content=Output.output)