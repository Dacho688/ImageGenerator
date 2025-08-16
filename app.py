import os
import gradio as gr
from transformers import HfEngine, Tool,CodeAgent,load_tool
from gradio_tools import StableDiffusionPromptGeneratorTool 
from streaming import stream_to_gradio
from huggingface_hub import login

# turn caching off
#client.headers["x-use-cache"] = "0"

#login
login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
#define llm engine
llm_engine = HfEngine("meta-llama/Llama-3.3-70B-Instruct")
#load tools
image_gen_tool = load_tool("huggingface-tools/text-to-image")
gradio_pg_tool = StableDiffusionPromptGeneratorTool()
pg_tool = Tool.from_gradio(gradio_pg_tool)
#create agent
agent = CodeAgent(
    tools=[pg_tool,image_gen_tool],
    llm_engine=llm_engine,
    additional_authorized_imports=[],
    max_iterations=10,
)
#base prompt 
base_prompt = """Improve the following prompt and generate an image.
Prompt:"""
#Main function to interact with streaming
def interact_with_agent(add_prompt):
    prompt = base_prompt
    if add_prompt and len(add_prompt) > 0:
        prompt += add_prompt
    else:
        prompt="There is no prompt made. Reply exactly with:'***ERROR: Please input a prompt.***'"
        
    messages = [gr.ChatMessage(role="assistant", content="⏳ _Generating image..._")]
    yield messages

    for msg in stream_to_gradio(agent, prompt):
        messages.append(msg)
        yield messages + [
            gr.ChatMessage(role="assistant", content="⏳ _Still processing..._")
        ]
    yield messages

#Gradio blocks and markdowns
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.yellow,
    )
) as demo:
    gr.Markdown("""# Image Generator (CodeAgent) 🖼️ 
                
I am an image generating Code Agent powered by Llama-3.3-70B-Instruct model. 
I come with two main tools, Gradio's Stable Diffusion prompt generator and an image generator powered by Runway's stable-diffusion-v1-5.
""")
    chatbot = gr.Chatbot(
        label="ImageBot",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png"
        ),
    )
    text_input = gr.Textbox(
        label="What image would you like to generate?"
    )
    submit = gr.Button("Run", variant="primary")
    submit.click(interact_with_agent, [text_input], [chatbot])

if __name__ == "__main__":
    demo.launch()