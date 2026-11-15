"""Minimal stand-in for gradio so the app wiring can be tested without installing it."""
import sys
import types


class _Comp:
    instances = []

    def __init__(self, *a, **kw):
        self.args, self.kw, self.events = a, kw, {}
        _Comp.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):  # .click / .change / .queue / .launch ...
        def handler(*a, **kw):
            self.events.setdefault(name, []).append((a, kw))
            return self
        return handler


def install():
    gr = types.ModuleType("gradio")
    for n in ["Blocks", "Tabs", "Tab", "Row", "Group", "Accordion", "Markdown", "File", "Button", "Checkbox", "CheckboxGroup",
              "Dataframe", "Dropdown", "Radio", "Plot", "Gallery", "Textbox", "State", "Number"]:
        setattr(gr, n, type(n, (_Comp,), {}))
    gr.update = lambda **kw: {"__update__": True, **kw}
    sys.modules["gradio"] = gr
    return gr
