"""Minimal stand-in for streamlit so streamlit_app.py can be run and tested without installing Streamlit.

Widgets return st.session_state[key] if set (as in Streamlit) or their default. A test "clicks" a button by
adding its key to CLICKS before running the script; st.rerun() restarts the script, like Streamlit does.
"""
import sys
import types


class Rerun(Exception):
    pass


class _State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


class _CM:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def install():
    st = types.ModuleType("streamlit")
    st.session_state = _State()
    st.CLICKS = set()
    st.LOG = []               # (kind, payload) of everything shown
    st.DOWNLOADS = []
    st.Rerun = Rerun

    def _log(kind):
        def f(*a, **kw):
            st.LOG.append((kind, a[0] if a else kw))
        return f

    for name in ("set_page_config", "title", "header", "subheader", "markdown", "caption", "info", "warning", "error",
                 "success", "write", "divider", "dataframe", "image", "code"):
        setattr(st, name, _log(name))

    def pyplot(fig, **kw):
        st.LOG.append(("pyplot", fig))
    st.pyplot = pyplot

    def _val(key, default):
        return st.session_state[key] if key is not None and key in st.session_state else default

    def file_uploader(label, type=None, key=None, **kw):
        return _val(key, None)

    def change(key, value):
        """Test helper: the user changes a widget; its on_change callback runs as in Streamlit."""
        st.session_state[key] = value
        cb = st.CALLBACKS.get(key)
        if cb:
            cb()
    st.change = change
    st.CALLBACKS = {}

    def _reg(key, kw):
        if key and kw.get("on_change"):
            st.CALLBACKS[key] = kw["on_change"]

    def checkbox(label, value=False, key=None, **kw):
        _reg(key, kw)
        v = _val(key, value)
        if key:
            st.session_state[key] = v
        return v

    def radio(label, options, index=0, format_func=str, key=None, **kw):
        _reg(key, kw)
        opts = list(options)
        v = _val(key, opts[index] if index is not None and opts else None)
        assert v in opts, f"radio {label!r}: {v!r} not in options"
        if key:
            st.session_state[key] = v
        return v

    def selectbox(label, options, index=0, format_func=str, key=None, **kw):
        opts = list(options)
        v = _val(key, opts[index] if opts else None)
        assert v in opts, f"selectbox {label!r}: {v!r} not in options"
        if key:
            st.session_state[key] = v
        for o in opts:
            format_func(o)       # labels must be computable
        return v

    def multiselect(label, options, default=None, format_func=str, key=None, **kw):
        _reg(key, kw)
        opts = list(options)
        for d in default or []:
            assert d in opts, f"multiselect {label!r}: default {d!r} not in options"
        v = list(_val(key, list(default or [])))
        assert all(x in opts for x in v), f"multiselect {label!r}: value not in options"
        if key:
            st.session_state[key] = v
        for o in opts:
            format_func(o)
        return v

    def text_input(label, value="", key=None, **kw):
        v = _val(key, value)
        if key:
            st.session_state[key] = v
        return v

    def text_area(label, value=None, key=None, **kw):
        v = _val(key, value if value is not None else "")
        if key:
            st.session_state[key] = v
        return v

    def data_editor(df, key=None, **kw):
        return _val(key, df)

    def button(label, key=None, on_click=None, disabled=False, **kw):
        if key in st.CLICKS:
            st.CLICKS.discard(key)
            assert not disabled, f"button {label!r} clicked while disabled"
            if on_click:
                on_click()
            return True
        return False

    def download_button(label, data=None, file_name=None, **kw):
        st.DOWNLOADS.append((file_name, data))
        return False

    def expander(*a, **kw):
        return _CM()

    def popover(*a, **kw):
        return _CM()

    def spinner(*a, **kw):
        return _CM()

    def tabs(names):
        return [_CM() for _ in names]

    def columns(spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [_CM() for _ in range(n)]

    def rerun():
        raise Rerun()

    def _logged(f):
        def w(*a, **kw):
            st.LOG.append(("widget", a[0] if a else kw.get("label")))
            return f(*a, **kw)
        w.__name__ = f.__name__
        return w

    for f in (file_uploader, checkbox, radio, selectbox, multiselect, text_input, text_area, data_editor, button, download_button):
        setattr(st, f.__name__, _logged(f))
    for f in (expander, popover, spinner, tabs, columns, rerun):
        setattr(st, f.__name__, f)
    cc = types.SimpleNamespace(SelectboxColumn=lambda *a, **kw: None)
    st.column_config = cc
    sys.modules["streamlit"] = st
    return st


def run(path, st, max_reruns=5):
    """Run the app script like Streamlit does, following st.rerun() calls."""
    import runpy
    for _ in range(max_reruns):
        st.LOG.clear()
        st.DOWNLOADS.clear()
        try:
            runpy.run_path(path, run_name="__main__")
            return
        except Rerun:
            continue
    raise RuntimeError("too many reruns")
