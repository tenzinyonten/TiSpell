"""Gradio demo for the BDRC/tibetan-byt5-v12b spell checker."""
import gradio as gr

from tibetan_spellcheck import MODEL_ID, correct

# Each example contains an error for the model to fix.
EXAMPLES = [
    ["བཀྲ་ཤིས་ཀྱིས་དཔེ་དེབ་གསར་པ་ཞིགཉོས་བྱུང་།"],
    ["གཞོན་སྐྱེས་རྣམས་སློབ་གྲྭ་ཁག་དུ་འགྲོ་བཞིན་ཡོད།"],
    ["བོད་གྱི་རིག་གཞུང་ནི་ལོ་ངོ་སྟོང་ཕྲག་མང་པོའི་རིང་ལ་དར་ཞིང་རྒྱས།"],
    ["རི་མོ་འདི་ནི་ཤིན་ཏུམཛེས་པོ་ཞིག་འདུག"],
    ["ཡི་གེ་འདི་དག་གསལ་པོར་ཀློགས།"],
]


def correct_text(text, add_shad_space, fix_particles):
    return correct([text], add_shad_space=add_shad_space,
                   fix_particles=fix_particles)[0]


text_in = gr.Textbox(label="Tibetan text (with errors)", lines=6)

demo = gr.Interface(
    fn=correct_text,
    inputs=[
        text_in,
        gr.Checkbox(label="Add space after punctuation (།)", value=False),
        gr.Checkbox(label="Apply rule-based particle fixes (experimental)", value=False),
    ],
    outputs=gr.Textbox(label="Corrected text", lines=6),
    title="Tibetan Spell Checker",
    description=(
        "Paste Tibetan text with spelling errors and get a corrected version "
        f"(model: {MODEL_ID}). Click an example to try it."
    ),
)

with demo:
    gr.Examples(examples=EXAMPLES, inputs=[text_in])

if __name__ == "__main__":
    demo.launch(share=True)
