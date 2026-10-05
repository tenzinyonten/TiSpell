"""Gradio demo for the BDRC/tibetan-byt5-v12b spell checker."""
import gradio as gr

from correct import MODEL_ID, correct

EXAMPLES = [
    ["བོད་ཀྱི་སྐད་ཡིག་ནི་ཧ་ཅང་ཡག་པོ་ཡིན།"],
    ["ང་ཚོས་དགེ་བའི་ལས་ལ་འབད་དགོས་ཀྱི་ཡོད།"],
    ["དེ་རིང་གནམ་གཤིས་ཧ་ཅང་བཟང་པོ་འདུག།"],
]


def correct_text(text):
    return correct([text])[0]


demo = gr.Interface(
    fn=correct_text,
    inputs=gr.Textbox(label="Input Tibetan text", lines=6),
    outputs=gr.Textbox(label="Corrected text", lines=6),
    examples=EXAMPLES,
    title="Tibetan spell checker",
    description=f"Model: {MODEL_ID}",
)

if __name__ == "__main__":
    demo.launch(share=True)
