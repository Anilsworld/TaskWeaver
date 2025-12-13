import io
import json
import types
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

from injector import inject

from taskweaver.llm.util import ChatMessageType
from taskweaver.logging import TelemetryLogger
from taskweaver.memory import Attachment, Post
from taskweaver.memory.attachment import AttachmentType
from taskweaver.module.event_emitter import PostEventProxy, SessionEventEmitter
from taskweaver.module.tracing import Tracing
from taskweaver.utils import json_parser


class PostTranslator:
    """
    PostTranslator is used to parse the output of the LLM or convert it to a Post object.
    The core function is post_to_raw_text and raw_text_to_post.
    """

    @inject
    def __init__(
        self,
        logger: TelemetryLogger,
        tracing: Tracing,
        event_emitter: SessionEventEmitter,
    ):
        self.logger = logger
        self.tracing = tracing
        self.event_emitter = event_emitter

    def raw_text_to_post(
        self,
        llm_output: Iterable[ChatMessageType],
        post_proxy: PostEventProxy,
        early_stop: Optional[Callable[[Union[AttachmentType, Literal["message", "send_to"]], str], bool]] = None,
        validation_func: Optional[Callable[[Post], None]] = None,
        use_v2_parser: bool = True,
    ) -> None:
        """
        Convert the raw text output from LLM to a Post object.
        """

        def stream_filter(s: Iterable[ChatMessageType]) -> Iterator[str]:
            full_llm_content = ""
            try:
                for c in s:
                    full_llm_content += c["content"]
                    yield c["content"]
            finally:
                if isinstance(s, types.GeneratorType):
                    try:
                        s.close()
                    except GeneratorExit:
                        pass
                output_size = self.tracing.count_tokens(full_llm_content)
                self.tracing.set_span_attribute("output_size", output_size)
                self.tracing.add_prompt_size(
                    size=output_size,
                    labels={
                        "direction": "output",
                    },
                )
                self.logger.info(f"LLM output: {full_llm_content}")

        value_buf: str = ""
        filtered_stream = stream_filter(llm_output)
        parser_stream = (
            self.parse_llm_output_stream_v2(filtered_stream)
            if use_v2_parser
            else self.parse_llm_output_stream(filtered_stream)
        )
        # parser_stream = self.parse_llm_output("".join([c["content"] for c in llm_output]))
        cur_attachment: Optional[Attachment] = None
        try:
            for type_str, value, is_end in parser_stream:
                value_buf += value
                type: Optional[AttachmentType] = None
                if type_str == "message":
                    post_proxy.update_message(value_buf, is_end=is_end)
                    value_buf = ""
                elif type_str == "send_to":
                    if is_end:
                        post_proxy.update_send_to(value_buf)  # type: ignore
                        value_buf = ""
                    else:
                        # collect the whole content before updating post
                        pass
                else:
                    try:
                        type = AttachmentType(type_str)
                        if cur_attachment is not None:
                            assert type == cur_attachment.type
                        cur_attachment = post_proxy.update_attachment(
                            value_buf,
                            type,
                            id=(cur_attachment.id if cur_attachment is not None else None),
                            is_end=is_end,
                        )
                        value_buf = ""
                        if is_end:
                            cur_attachment = None
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to parse attachment: {type_str}-{value_buf} due to {str(e)}",
                        )
                        continue
                parsed_type = (
                    type
                    if type is not None
                    else "message"
                    if type_str == "message"
                    else "send_to"
                    if type_str == "send_to"
                    else None
                )
                assert parsed_type is not None, f"Invalid type: {type_str}"

                # check whether parsing should be triggered prematurely when each key parsing is finished
                if is_end and early_stop is not None and early_stop(parsed_type, value):
                    break
        finally:
            if isinstance(parser_stream, types.GeneratorType):
                try:
                    parser_stream.close()
                except GeneratorExit:
                    pass

        if validation_func is not None:
            validation_func(post_proxy.post)

    def post_to_raw_text(
        self,
        post: Post,
        content_formatter: Callable[[Attachment], str] = lambda x: x.content,
        if_format_message: bool = True,
        if_format_send_to: bool = True,
        ignored_types: Optional[List[AttachmentType]] = None,
    ) -> str:
        """
        Convert a Post object to raw text in the format of LLM output.
        """
        if ignored_types is None:
            ignored_types = []
        ignored_types.append(AttachmentType.shared_memory_entry)

        structured_llm: Dict[str, str] = {}
        for attachment in post.attachment_list:
            if ignored_types is not None and attachment.type in ignored_types:
                continue
            if attachment.type.value not in structured_llm:
                structured_llm[attachment.type.value] = content_formatter(attachment)
            else:
                # append the content of the same type of attachment
                structured_llm[attachment.type.value] += f"\n{content_formatter(attachment)}"
        if if_format_send_to:
            structured_llm["send_to"] = post.send_to
        if if_format_message:
            structured_llm["message"] = post.message
        structured_llm_text = json.dumps({"response": structured_llm})
        return structured_llm_text

    def parse_llm_output(self, llm_output: str) -> Iterator[Tuple[str, str, bool]]:
        try:
            parsed_json = json.loads(llm_output)
            
            # CRITICAL FIX: Azure OpenAI API 2025-01-01-preview with json_schema wraps responses
            # Check if response is wrapped in schema structure: {"type":"object","properties":{"response":{...}}}
            if isinstance(parsed_json, dict) and "type" in parsed_json and "properties" in parsed_json:
                # Unwrap: extract actual data from properties.response
                if "response" in parsed_json.get("properties", {}):
                    structured_llm_output: Any = parsed_json["properties"]["response"]
                else:
                    # Fallback: if no "response" key, use properties directly
                    structured_llm_output: Any = parsed_json["properties"]
            elif "response" in parsed_json:
                # Standard format: {"response": {...}}
                structured_llm_output: Any = parsed_json["response"]
            else:
                # No wrapping at all - use as-is
                structured_llm_output: Any = parsed_json
            
            kv_pairs = []
            assert isinstance(
                structured_llm_output,
                dict,
            ), "LLM output should be a dict object"
            for key in structured_llm_output:
                if isinstance(structured_llm_output[key], str):
                    kv_pairs.append((key, structured_llm_output[key], True))
                else:
                    raise AssertionError(
                        f"Invalid LLM output format: {structured_llm_output[key]}",
                    )
            return kv_pairs  # type: ignore
        except (json.JSONDecodeError, AssertionError) as e:
            self.logger.error(
                f"Failed to parse LLM output due to {str(e)}. LLM output:\n {llm_output}",
            )
            raise e

    def parse_llm_output_stream(
        self,
        llm_output: Iterator[str],
    ) -> Iterator[Tuple[str, str, bool]]:
        import ijson

        class StringIteratorIO(io.TextIOBase):
            def __init__(self, iter: Iterator[str]):
                self._iter = iter
                self._left: str = ""

            def readable(self):
                return True

            def _read1(self, n: Optional[int] = None):
                while not self._left:
                    try:
                        self._left = next(self._iter)
                    except StopIteration:
                        break
                ret = self._left[:n]
                self._left = self._left[len(ret) :]
                return ret

            def read(self, n: Optional[int] = None):
                l: List[str] = []
                if n is None or n < 0:
                    while True:
                        m = self._read1()
                        if not m:
                            break
                        l.append(m)
                else:
                    while n > 0:
                        m = self._read1(n)
                        if not m:
                            break
                        n -= len(m)
                        l.append(m)
                return "".join(l)

        json_data_stream = StringIteratorIO(llm_output)
        # use small buffer to get parse result as soon as acquired from LLM
        parser = ijson.parse(json_data_stream, buf_size=5)

        cur_type: Optional[str] = None
        cur_content: Optional[str] = None
        try:
            for prefix, event, value in parser:
                # CRITICAL FIX: Handle ALL Azure wrapping variants
                # 1. Normal: {"response": {...}}
                # 2. Azure with response: {"type":"object","properties":{"response":{...}}}
                # 3. Azure without response: {"type":"object","properties":{...}}  ← NEW!
                
                # Check for map keys in all 3 formats
                if event == "map_key":
                    if prefix == "response":
                        # Format 1: response.key
                        cur_type = value
                    elif prefix == "properties.response":
                        # Format 2: properties.response.key
                        cur_type = value
                    elif prefix == "properties":
                        # Format 3: properties.key (NEW!)
                        cur_type = value
                
                # Check for string values in all 3 formats
                if cur_type and event == "string":
                    value_prefixes = [
                        f"response.{cur_type}",              # Format 1
                        f"properties.response.{cur_type}",   # Format 2
                        f"properties.{cur_type}",            # Format 3 (NEW!)
                    ]
                    if prefix in value_prefixes:
                        cur_content = value

                if cur_type is not None and cur_content is not None:
                    yield cur_type, cur_content, True
                    cur_type, cur_content = None, None
        except ijson.JSONError as e:
            self.logger.warning(
                f"Failed to parse LLM output stream due to JSONError: {str(e)}",
            )
        finally:
            if isinstance(llm_output, types.GeneratorType):
                try:
                    llm_output.close()
                except GeneratorExit:
                    pass

    def parse_llm_output_stream_v2(
        self,
        llm_output: Iterator[str],
    ) -> Iterator[Tuple[str, str, bool]]:
        parser = json_parser.parse_json_stream(
            llm_output,
            skip_after_root=True,
            include_all_values=True,
            skip_ws=True,
        )
        # CRITICAL FIX: Support ALL 3 Azure wrapping variants
        # 1. Normal: {"response": {...}} → prefix ".response"
        # 2. Azure with response: {"type":"object","properties":{"response":{...}}} → prefix ".properties.response"
        # 3. Azure without response: {"type":"object","properties":{...}} → prefix ".properties" (NEW!)
        root_element_prefixes = [".response", ".properties.response", ".properties"]

        cur_type: Optional[str] = None
        try:
            for ev in parser:
                # Check if prefix matches any of the supported formats
                matched_prefix = None
                for prefix in root_element_prefixes:
                    if ev.prefix == prefix or ev.prefix.startswith(prefix + "."):
                        matched_prefix = prefix
                        break
                
                if not matched_prefix:
                    continue
                
                if ev.prefix == matched_prefix and ev.event == "map_key" and ev.is_end:
                    cur_type = ev.value
                    yield cur_type, "", False
                elif ev.prefix == f"{matched_prefix}.{cur_type}" and ev.event == "string":
                    yield cur_type, ev.value_str, ev.is_end
                elif ev.prefix == f"{matched_prefix}.{cur_type}" and ev.event == "number":
                    yield cur_type, ev.value_str, ev.is_end
                elif ev.prefix == f"{matched_prefix}.{cur_type}" and ev.event == "boolean":
                    yield cur_type, ev.value_str, ev.is_end
                elif ev.prefix == f"{matched_prefix}.{cur_type}" and ev.event == "null":
                    yield cur_type, "", True
                elif ev.prefix == f"{matched_prefix}.{cur_type}" and ev.event == "start_map":
                    self.logger.warning(f"Start map in property: {matched_prefix}.{cur_type}")
                elif ev.prefix == f"{matched_prefix}.{cur_type}" and ev.event == "end_map":
                    yield cur_type, json.dumps(ev.value), True
                elif ev.prefix == f"{matched_prefix}.{cur_type}" and ev.event == "start_array":
                    self.logger.warning(f"Start array in property: {matched_prefix}.{cur_type}")
                elif ev.prefix == f"{matched_prefix}.{cur_type}" and ev.event == "end_array":
                    yield cur_type, json.dumps(ev.value), True

        except json_parser.StreamJsonParserError as e:
            self.logger.warning(
                f"Failed to parse LLM output stream due to JSONError: {str(e)}",
            )

        finally:
            if isinstance(parser, types.GeneratorType):
                try:
                    parser.close()
                except GeneratorExit:
                    pass
