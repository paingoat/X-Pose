"""Compatibility shims for Gradio 4.44.x + newer pydantic/fastapi JSON schemas.

gradio_client 1.3.0 assumes every JSON-schema node is a dict. Newer pydantic can
emit boolean schemas (e.g. additionalProperties: true), which crashes get_api_info()
with: TypeError: argument of type 'bool' is not iterable

Upstream: https://github.com/gradio-app/gradio/issues/11722
"""

from __future__ import annotations


def patch_gradio_client_schema_bool() -> None:
    import gradio_client.utils as client_utils

    if getattr(client_utils, "_xpose_bool_schema_patched", False):
        return

    _orig_get_type = client_utils.get_type
    _orig_json = client_utils._json_schema_to_python_type

    def get_type(schema):
        if isinstance(schema, bool):
            return "Any" if schema else "Any"
        return _orig_get_type(schema)

    def _json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            # true => unrestricted object values; false => no additional props
            return "Any"
        return _orig_json(schema, defs)

    client_utils.get_type = get_type
    client_utils._json_schema_to_python_type = _json_schema_to_python_type
    client_utils._xpose_bool_schema_patched = True
