import os
from .ui import build_app

app = build_app()
app.queue()
app.launch(share=os.getenv("METAPY_SHARE", "true").strip().lower() not in {"0", "false", "no", "off"}, debug=True)
