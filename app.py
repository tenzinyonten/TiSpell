"""Gradio demo for the BDRC/tibetan-byt5-v12b spell checker."""
import gradio as gr

from tibetan_spellcheck import MODEL_ID, correct

# Each example contains an error for the model to fix.
EXAMPLES = [
    ["བཀྲ་ཤིས་ཀྱིས་དཔེ་དེབ་གསར་པ་ཞིགཉོས་བྱུང་།", False],
    ["གཞོན་སྐྱེས་རྣམས་སློབ་གྲྭ་ཁག་དུ་འགྲོ་བཞིན་ཡོད།", False],
    ["བོད་གྱི་རིག་གཞུང་ནི་ལོ་ངོ་སྟོང་ཕྲག་མང་པོའི་རིང་ལ་དར་ཞིང་རྒྱས།", False],
    ["རི་མོ་འདི་ནི་ཤིན་ཏུམཛེས་པོ་ཞིག་འདུག", False],
    ["ཡི་གེ་འདི་དག་གསལ་པོར་ཀློགས།", False],
]


def correct_text(text, add_shad_space):
    return correct([text], add_shad_space=add_shad_space)[0]


demo = gr.Interface(
    fn=correct_text,
    inputs=[
        gr.Textbox(label="Tibetan text (with errors)", lines=6),
        gr.Checkbox(label="Add space after punctuation (།)", value=False),
    ],
    outputs=gr.Textbox(label="Corrected text", lines=6),
    examples=EXAMPLES,
    title="Tibetan Spell Checker",
    description=(
        "Paste Tibetan text with spelling errors and get a corrected version "
        f"(model: {MODEL_ID}). Click an example to try it."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
