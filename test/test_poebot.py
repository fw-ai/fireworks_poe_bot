from typing import List
from fireworks_poe_bot.fw_poe_text_bot import (
    FireworksPoeTextBot,
)
from sse_starlette.sse import ServerSentEvent
from fastapi_poe.types import (
    QueryRequest,
    ProtocolMessage,
    PartialResponse,
    ErrorResponse,
)
from fastapi import Request

import unittest

from fireworks.client.api import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)


async def fake_chat_completion_acreate(
    model, *args, request_timeout=600, stream=False, **kwargs
):
    assert len(args) == 0
    messages = kwargs["messages"]

    expect_user = True
    last_role = None
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            assert i == 0, "System message not first message"
        elif msg["role"] == "user":
            assert expect_user, "User message not expected"
            expect_user = False
        elif msg["role"] == "assistant":
            assert not expect_user, "Assistant message not expected"
            expect_user = True
        last_role = msg["role"]

    assert last_role == "user"

    assert stream

    for resp in ["foo"]:
        yield ChatCompletionStreamResponse(
            id="",
            object="",
            created=0,
            model=model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0, delta=DeltaMessage(role="assistant", content=resp)
                )
            ],
        )


class TestFWPoeBot(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = ""
        self.environment = ""
        self.deployment = ""
        self.server_version = ""
        self.completion_async_method = fake_chat_completion_acreate
        self.bot = FireworksPoeTextBot(
            model=self.model,
            api_key="",
            environment=self.environment,
            deployment=self.deployment,
            server_version=self.server_version,
            allow_attachments=False,
            input_image_size=-1,
            prompt_truncate_len=1024,
            max_tokens=2048,
            system_prompt_override="",
            additional_args=None,
            chat_format=None,
            alpaca_instruction_msg=None,
            meta_response=None,
            completion_async_method=self.completion_async_method,
        )

    class FakeRequest(Request):
        def __init__(self):
            pass

    async def _test_with_query(self, query: List[ProtocolMessage]):
        query_request = QueryRequest(
            version="",
            type="query",
            query=query,
            user_id="",
            conversation_id="",
            message_id="",
            http_request=self.FakeRequest(),
        )
        resp_fragments = []
        async for resp in self.bot.get_response(query_request):
            if isinstance(resp, ErrorResponse):
                self.fail(resp.text)
            elif isinstance(resp, PartialResponse):
                resp_fragments.append(resp.text)
            elif isinstance(resp, ServerSentEvent):
                assert resp.event == "done"
        return "".join(resp_fragments)

    async def test_empty_query(self):
        with self.assertRaisesRegex(AssertionError, "Empty query"):
            await self._test_with_query([])

    async def test_single_query(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="user", content="hello"),
            ]
        )
        self.assertEqual(resp, "foo")

    async def test_single_req_response(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="user", content="hello"),
                ProtocolMessage(role="bot", content="foo"),
                ProtocolMessage(role="user", content="bar"),
            ]
        )
        self.assertEqual(resp, "foo")

    async def test_system_prompt(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="system", content="hello"),
                ProtocolMessage(role="user", content="foo"),
            ]
        )
        self.assertEqual(resp, "foo")

    async def test_system_prompt_req_response(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="system", content="hello"),
                ProtocolMessage(role="user", content="foo"),
                ProtocolMessage(role="bot", content="bar"),
                ProtocolMessage(role="user", content="baz"),
            ]
        )
        self.assertEqual(resp, "foo")

    async def test_no_initial_user_msg(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="bot", content="hello"),
                ProtocolMessage(role="user", content="foo"),
            ]
        )
        self.assertEqual(resp, "foo")

    async def test_duplicate_user_msgs(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="user", content="hello"),
                ProtocolMessage(role="user", content="foo"),
            ]
        )
        self.assertEqual(resp, "foo")

    async def test_duplicate_assistant_msgs(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="user", content="hello"),
                ProtocolMessage(role="bot", content="foo"),
                ProtocolMessage(role="bot", content="bar"),
            ]
        )
        self.assertEqual(resp, "foo")

    async def test_no_final_user_msg(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="user", content="hello"),
                ProtocolMessage(role="bot", content="foo"),
            ]
        )
        self.assertEqual(resp, "foo")

    async def test_multiple_system_msgs(self):
        resp = await self._test_with_query(
            [
                ProtocolMessage(role="system", content="hello"),
                ProtocolMessage(role="system", content="foo"),
                ProtocolMessage(role="user", content="bar"),
            ]
        )
        self.assertEqual(resp, "foo")


if __name__ == "__main__":
    unittest.main()
