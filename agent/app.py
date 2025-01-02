import gradio as gr
from Chatbot import main
import asyncio

async def process_query(message):
    response = await main()
    return response

demo = gr.Interface(
    fn=lambda x: asyncio.run(process_query(x)),
    inputs=gr.Textbox(label="User Input", placeholder="Nhập câu hỏi tại đây..."),
    outputs=gr.Textbox(label="Agent Response"),
    title="Flight Assistant",
    description="Hãy nhập câu hỏi của bạn bên dưới:"
)

if __name__ == "__main__":
    demo.launch()