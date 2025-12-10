"""
title: Auto Code Analysis
author: @nokodo
description: execute Python code to perform any analysis, file operations, or data processing tasks.
author_email: nokodo@nokodo.net
author_url: https://nokodo.net
funding_url: https://ko-fi.com/nokodo
repository_url: https://nokodo.net/github/open-webui-extensions
version: 1.0.0
required_open_webui_version: >= 0.6.0
requirements: aiohttp, websockets, e2b_code_interpreter
license: see extension documentation file `auto_code_analysis.md` (License section) for the licensing terms.
"""

import base64
import datetime
import io
import json
import logging
import mimetypes
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

import aiohttp
from e2b_code_interpreter import AsyncSandbox
from open_webui.models.files import FileForm, Files
from open_webui.storage.provider import Storage
from open_webui.utils.code_interpreter import JupyterCodeExecuter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ==========================================
# Helper Functions
# ==========================================


async def emit_status(
    description: str,
    emitter: Any,
    status: Literal["in_progress", "complete", "error"] = "complete",
    done: bool = False,
    error: bool = False,
):
    if not emitter:
        return
    await emitter(
        {
            "type": "status",
            "data": {
                "description": description,
                "done": done,
                "error": error,
                "status": status,
            },
        }
    )


async def emit_files(emitter: Any, files: list[dict]):
    """Emit files to the frontend for display/download."""
    if not emitter or not files:
        return
    await emitter(
        {
            "type": "files",
            "data": {
                "files": files,
            },
        }
    )


# ==========================================
# Abstract Base Classes
# ==========================================


class ExecutionResult(BaseModel):
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    results: list[Any] = []
    error: Optional[dict[str, Any]] = None
    files: list[dict[str, Any]] = []  # Metadata of generated files
    sandbox_id: Optional[str] = None


class SandboxAdapter(ABC):
    @abstractmethod
    async def start(self, sandbox_id: Optional[str] = None):
        """Initialize or resume the sandbox environment."""
        pass

    @abstractmethod
    async def run_code(self, code: str, timeout: int = 60) -> ExecutionResult:
        """Execute code in the sandbox."""
        pass

    @abstractmethod
    async def upload_file(self, filename: str, content: bytes):
        """Upload a file to the sandbox working directory."""
        pass

    @abstractmethod
    async def download_file(self, filename: str) -> Optional[bytes]:
        """Download a file from the sandbox working directory."""
        pass

    @abstractmethod
    async def list_files(self, path: str = ".") -> list[str]:
        """list files in the sandbox working directory."""
        pass

    @abstractmethod
    async def stop(self):
        """Clean up resources or pause the sandbox."""
        pass


# ==========================================
# Jupyter Implementation
# ==========================================


class JupyterFileHandler:
    def __init__(self, session: aiohttp.ClientSession, base_url: str, token: str = ""):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _get_headers(self):
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    async def create_directory(self, path: str):
        url = f"{self.base_url}/api/contents/{path}"
        data = {"type": "directory"}
        async with self.session.put(
            url, json=data, headers=self._get_headers()
        ) as response:
            if response.status not in (200, 201):
                # It's okay if it already exists, usually
                pass

    async def delete_path(self, path: str):
        """Delete a file or directory recursively."""
        url = f"{self.base_url}/api/contents/{path}"
        async with self.session.get(url, headers=self._get_headers()) as response:
            if response.status == 404:
                return
            if response.status == 200:
                data = await response.json()
                if data.get("type") == "directory":
                    for item in data.get("content", []):
                        await self.delete_path(f"{path}/{item['name']}")

        async with self.session.delete(url, headers=self._get_headers()) as response:
            if response.status != 404:
                # response.raise_for_status()
                pass

    async def upload_file(self, filepath: str, content: bytes):
        url = f"{self.base_url}/api/contents/{filepath}"
        b64_content = base64.b64encode(content).decode("utf-8")
        data = {
            "content": b64_content,
            "format": "base64",
            "type": "file",
        }
        async with self.session.put(
            url, json=data, headers=self._get_headers()
        ) as response:
            response.raise_for_status()

    async def download_file(self, filepath: str) -> Optional[bytes]:
        url = f"{self.base_url}/api/contents/{filepath}"
        params = {"format": "base64"}
        async with self.session.get(
            url, params=params, headers=self._get_headers()
        ) as response:
            if response.status == 404:
                return None
            response.raise_for_status()
            data = await response.json()
            if data.get("format") == "base64":
                return base64.b64decode(data["content"])
            else:
                return data["content"].encode("utf-8")

    async def list_files(self, path: str = "") -> list[str]:
        url = f"{self.base_url}/api/contents/{path}"
        async with self.session.get(url, headers=self._get_headers()) as response:
            if response.status == 404:
                return []
            response.raise_for_status()
            data = await response.json()
            if data.get("type") != "directory":
                return []
            return [
                item["name"]
                for item in data.get("content", [])
                if item["type"] == "file"
            ]

    async def list_contents(self, path: str) -> list[dict[str, Any]]:
        """list all contents of a directory with metadata."""
        url = f"{self.base_url}/api/contents/{path}"
        async with self.session.get(url, headers=self._get_headers()) as response:
            if response.status == 404:
                return []
            response.raise_for_status()
            data = await response.json()
            if data.get("type") != "directory":
                return []
            return data.get("content", [])


class JupyterSandbox(SandboxAdapter):
    def __init__(
        self, url: str, auth: str, token: str, password: str, retention_hours: int = 24
    ):
        self.url = url
        self.auth = auth
        self.token = token
        self.password = password
        self.retention_hours = retention_hours
        self.executor = None
        self.file_handler = None
        self.session_id = None
        self.session_path = None

    async def cleanup_old_sessions(self):
        if self.retention_hours <= 0 or not self.file_handler:
            return

        try:
            contents = await self.file_handler.list_contents("work/sessions")
            now = datetime.datetime.now(datetime.timezone.utc)

            for item in contents:
                if item["type"] == "directory":
                    # item["last_modified"] is like "2023-10-27T10:00:00.000000Z"
                    last_modified_str = item.get("last_modified", "")
                    if not last_modified_str:
                        continue

                    try:
                        # Handle Z for UTC
                        if last_modified_str.endswith("Z"):
                            last_modified_str = last_modified_str[:-1] + "+00:00"
                        last_modified = datetime.datetime.fromisoformat(
                            last_modified_str
                        )
                    except ValueError:
                        continue

                    age = now - last_modified
                    if age > datetime.timedelta(hours=self.retention_hours):
                        await self.file_handler.delete_path(
                            f"work/sessions/{item['name']}"
                        )
        except Exception as e:
            logger.warning(f"Failed to cleanup old sessions: {e}")

    async def start(self, sandbox_id: Optional[str] = None):
        # For Jupyter, we use a session directory concept.
        # If sandbox_id is provided, we try to reuse that directory.
        self.session_id = sandbox_id or str(uuid.uuid4())
        self.session_path = f"work/sessions/{self.session_id}"

        token = self.token if self.auth == "token" else ""
        password = self.password if self.auth == "password" else ""

        # We initialize the executor but don't run code yet
        self.executor = JupyterCodeExecuter(
            base_url=self.url,
            code="",
            token=token,
            password=password,
            timeout=60,  # Default, will be overridden in run_code
        )

        # We need to sign in to get the session
        await self.executor.sign_in()
        await self.executor.init_kernel()

        self.file_handler = JupyterFileHandler(
            self.executor.session, self.url, token=token
        )

        # Ensure directories exist
        try:
            await self.file_handler.create_directory("work")
        except Exception:
            pass
        await self.file_handler.create_directory("work/sessions")

        # Cleanup old sessions
        await self.cleanup_old_sessions()

        await self.file_handler.create_directory(self.session_path)

    async def run_code(self, code: str, timeout: int = 60) -> ExecutionResult:
        if not self.executor:
            raise RuntimeError("Sandbox not started")

        self.executor.timeout = timeout

        # Wrap code to run in isolated directory
        wrapped_code = f"""
import os
try:
    os.chdir('{self.session_path}')
except FileNotFoundError:
    os.makedirs('{self.session_path}', exist_ok=True)
    os.chdir('{self.session_path}')

{code}
"""
        self.executor.code = wrapped_code
        result = await self.executor.run()

        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            results=[result.result] if result.result else [],
            sandbox_id=self.session_id,
        )

    async def upload_file(self, filename: str, content: bytes):
        if not self.file_handler:
            raise RuntimeError("Sandbox not started")
        await self.file_handler.upload_file(f"{self.session_path}/{filename}", content)

    async def download_file(self, filename: str) -> Optional[bytes]:
        if not self.file_handler:
            raise RuntimeError("Sandbox not started")
        return await self.file_handler.download_file(f"{self.session_path}/{filename}")

    async def list_files(self, path: str = ".") -> list[str]:
        if not self.file_handler or not self.session_path:
            raise RuntimeError("Sandbox not started")
        # path is relative to session_path
        target_path = self.session_path
        if path != ".":
            target_path = f"{self.session_path}/{path}"
        return await self.file_handler.list_files(target_path)

    async def stop(self):
        # We do not delete the session directory to allow for persistence across turns.
        # However, we should ensure the kernel is shut down if we are not using a persistent kernel manager.
        # The current JupyterCodeExecuter implementation in Open WebUI might not support persistent kernels easily
        # without keeping the object alive. Since this tool is stateless between calls, the kernel will likely die.
        # But the files will remain in the session directory.
        pass


# ==========================================
# E2B Implementation
# ==========================================


class E2BSandbox(SandboxAdapter):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sandbox = None

    async def start(self, sandbox_id: Optional[str] = None):
        if sandbox_id:
            try:
                self.sandbox = await AsyncSandbox.connect(
                    sandbox_id, api_key=self.api_key
                )
            except Exception as e:
                logger.warning(f"Failed to resume sandbox {sandbox_id}: {e}")
                self.sandbox = None

        if not self.sandbox:
            self.sandbox = await AsyncSandbox.create(
                template="nokodo-ai-code-interpreter-v1", api_key=self.api_key
            )

    async def run_code(self, code: str, timeout: int = 60) -> ExecutionResult:
        if not self.sandbox:
            raise RuntimeError("Sandbox not started")

        execution = await self.sandbox.run_code(code, timeout=timeout)

        # Extract images and save to sandbox so they are picked up by list_files
        images = await self.extract_images_from_execution(execution.results)
        for filename, content, mime_type in images:
            await self.sandbox.files.write(filename, content)

        stdout = ""
        if execution.logs.stdout:
            out = execution.logs.stdout
            if isinstance(out, list):
                stdout = "".join(out)
            else:
                stdout = str(out)

        stderr = ""
        if execution.logs.stderr:
            err = execution.logs.stderr
            if isinstance(err, list):
                stderr = "".join(err)
            else:
                stderr = str(err)

        results = []
        for result in execution.results:
            # We handle rich results (images) in the main Tool class by checking files/results
            # But here we just return the text representation or the object
            if hasattr(result, "text") and result.text:
                results.append(result.text)
            # Images are handled via side-effects in E2B usually, or we need to extract them here.
            # The original E2B tool extracted images from `execution.results` and saved them as files.
            # We should probably do that here and return them as "files" in the result.
            # But `upload_file` / `download_file` is for disk files.
            # Let's handle the extraction in the Tool logic or here?
            # Better here to keep the Tool generic.

        # We will handle image extraction in the main loop by checking the sandbox state or results
        # The original E2B tool did it in `execute_python_code`.

        error = None
        if execution.error:
            error = {
                "name": execution.error.name,
                "value": execution.error.value,
                "traceback": execution.error.traceback,
            }

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            results=results,
            error=error,
            sandbox_id=self.sandbox.sandbox_id,
        )

    async def upload_file(self, filename: str, content: bytes):
        if not self.sandbox:
            raise RuntimeError("Sandbox not started")
        await self.sandbox.files.write(filename, content)

    async def download_file(self, filename: str) -> Optional[bytes]:
        if not self.sandbox:
            raise RuntimeError("Sandbox not started")
        try:
            content = await self.sandbox.files.read(filename)
            if isinstance(content, str):
                return content.encode("utf-8")
            return content
        except Exception:
            return None

    async def list_files(self, path: str = ".") -> list[str]:
        if not self.sandbox:
            raise RuntimeError("Sandbox not started")
        entries = await self.sandbox.files.list(path)
        return [f.name for f in entries]

    async def stop(self):
        if self.sandbox:
            await self.sandbox.beta_pause(api_key=self.api_key)

    # Special method for E2B to extract images from results
    async def extract_images_from_execution(self, execution_results):
        images = []
        for result in execution_results:
            formats = []
            if hasattr(result, "png") and result.png:
                formats.append(("png", result.png))
            if hasattr(result, "jpeg") and result.jpeg:
                formats.append(("jpeg", result.jpeg))
            if hasattr(result, "svg") and result.svg:
                formats.append(("svg", result.svg))
            if hasattr(result, "pdf") and result.pdf:
                formats.append(("pdf", result.pdf))

            for ext, data_b64 in formats:
                try:
                    data = base64.b64decode(data_b64)
                    filename = f"chart_{uuid.uuid4().hex[:8]}.{ext}"
                    images.append((filename, data, f"image/{ext}"))
                except Exception:
                    pass
        return images


# ==========================================
# Main Tool Class
# ==========================================


class Tools:
    class Valves(BaseModel):
        ENGINE: Literal["jupyter", "e2b"] = Field(
            default="jupyter",
            description="Execution engine to use: 'jupyter' (self-hosted) or 'e2b' (cloud sandbox).",
        )
        # Jupyter Settings
        JUPYTER_URL: str = Field(
            default="http://host.docker.internal:8888",
            description="Jupyter server URL (if using Jupyter engine)",
        )
        JUPYTER_AUTH: Literal["token", "password", "none"] = Field(
            default="token",
            description="Jupyter authentication method",
        )
        JUPYTER_TOKEN: str = Field(
            default="",
            description="Jupyter authentication token",
        )
        JUPYTER_PASSWORD: str = Field(
            default="",
            description="Jupyter password",
        )
        JUPYTER_RETENTION_HOURS: int = Field(
            default=72,
            description="Retention period for Jupyter session directories in hours (0 to disable cleanup).",
        )
        # E2B Settings
        E2B_API_KEY: str = Field(
            default="",
            description="E2B API Key (if using E2B engine)",
        )
        # Common Settings
        TIMEOUT: int = Field(
            default=60,
            description="Execution timeout in seconds",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": (
                        "Execute Python code in a notebook to perform calculations, data analysis, and any file operations like creating or editing files.\n"
                        "The notebook comes pre-installed with all the libraries you might need.\n"
                        "Your notebook session and environment automatically persists across this entire conversation, so you can build and iterate over multiple turns.\n"
                        "You can upload files to the execution environment and download any generated files after execution. All user-attached files to the current chat are automatically included in the execution environment."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute.",
                            },
                            "action_name": {
                                "type": "string",
                                "description": "Required. A short verb phrase describing what this run is doing (e.g., 'analyzing dataset', 'generating report', 'creating your presentation').",
                            },
                            "file_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of file IDs to include in the execution environment.",
                                "default": [],
                            },
                        },
                        "required": ["code", "action_name"],
                    },
                },
            }
        ]

    async def read_file_content(self, file_path: str) -> bytes:
        """Read file content from URL or local path."""
        if file_path.startswith("http"):
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path) as response:
                    response.raise_for_status()
                    return await response.read()
        else:
            with open(file_path, "rb") as f:
                return f.read()

    async def save_file_to_storage(
        self, filename: str, content: bytes, mime_type: Optional[str], user: Any
    ) -> dict[str, Any]:
        """Save a file to Open WebUI storage and return metadata."""
        file_id = str(uuid.uuid4())
        unique_filename = f"{file_id}_{filename}"

        if not mime_type:
            mime_type_guess, _ = mimetypes.guess_type(filename)
            mime_type = mime_type_guess or "application/octet-stream"

        # upload to storage
        file_io = io.BytesIO(content)
        _, storage_path = Storage.upload_file(
            file_io,
            unique_filename,
            {
                "OpenWebUI-User-Email": (user.get("email", "") if user else ""),
                "OpenWebUI-User-Id": (user.get("id", "") if user else ""),
                "OpenWebUI-User-Name": (user.get("name", "") if user else ""),
                "OpenWebUI-File-Id": file_id,
            },
        )

        # register in DB
        Files.insert_new_file(
            user["id"] if user else "unknown",
            FileForm(
                **{
                    "id": file_id,
                    "filename": unique_filename,
                    "path": storage_path,
                    "data": {},
                    "meta": {
                        "name": filename,
                        "content_type": mime_type,
                        "size": len(content),
                    },
                }
            ),
        )

        file_url = f"/api/v1/files/{file_id}/content"
        return {
            "filename": filename,
            "download_url": file_url,
            "type": mime_type,
            "size": len(content),
        }

    async def execute_python_code(
        self,
        code: str,
        action_name: Optional[str] = None,
        file_ids: list[str] = [],
        __event_emitter__: Any = None,
        __user__: Any = None,
        __files__: list[dict] = [],
        __messages__: list[dict] = [],
    ) -> str:
        """Execute Python code using the configured engine."""

        # 1. Setup and Validation
        action_label = (action_name or "").strip()
        if not action_label:
            return "Please set 'action_name' to a short description of the task."

        await emit_status(
            action_label,
            status="in_progress",
            emitter=__event_emitter__,
            done=False,
        )

        # 2. Determine Engine and Sandbox ID
        engine = self.valves.ENGINE
        sandbox_id = None

        # Try to find existing sandbox ID in conversation history
        if __messages__:
            for message in reversed(__messages__):
                if message.get("role") == "assistant" and message.get("content"):
                    # Look for <!-- sandbox_id: ... -->
                    match = re.search(r"<!-- sandbox_id: (.+?) -->", message["content"])
                    if match:
                        sandbox_id = match.group(1)
                        # Also try to find engine if possible, or assume current config
                        # Ideally we should store engine too: <!-- engine: ... -->
                        engine_match = re.search(
                            r"<!-- engine: (.+?) -->", message["content"]
                        )
                        if engine_match:
                            # If the previous turn used a different engine, we might want to respect it
                            # OR switch to the currently configured one.
                            # Switching engines with the same ID won't work.
                            # So if engine differs, we probably should ignore the ID.
                            prev_engine = engine_match.group(1)
                            if prev_engine != engine:
                                sandbox_id = None
                        break

                    # Legacy support for E2B specific tag
                    match_e2b = re.search(
                        r"<!-- e2b_sandbox_id: (.+?) -->", message["content"]
                    )
                    if match_e2b and engine == "e2b":
                        sandbox_id = match_e2b.group(1)
                        break

        # 3. Initialize Sandbox Adapter
        sandbox: SandboxAdapter
        try:
            if engine == "e2b":
                if not self.valves.E2B_API_KEY:
                    raise ValueError("E2B_API_KEY is not set.")
                sandbox = E2BSandbox(self.valves.E2B_API_KEY)
            elif engine == "jupyter":
                sandbox = JupyterSandbox(
                    self.valves.JUPYTER_URL,
                    self.valves.JUPYTER_AUTH,
                    self.valves.JUPYTER_TOKEN,
                    self.valves.JUPYTER_PASSWORD,
                    self.valves.JUPYTER_RETENTION_HOURS,
                )
            else:
                raise ValueError(f"Unknown engine: {engine}")

            await sandbox.start(sandbox_id)
        except Exception as e:
            logger.exception("Failed to start sandbox")
            await emit_status(
                "error starting analysis",
                status="error",
                emitter=__event_emitter__,
                done=True,
                error=True,
            )
            return f"Error starting sandbox: {str(e)}"

        try:
            # 4. Handle File Uploads
            # Add files attached to the message automatically
            if __files__:
                for file in __files__:
                    if file.get("id") and file["id"] not in file_ids:
                        file_ids.append(file["id"])

            uploaded_filenames = []
            if file_ids:
                for file_id in file_ids:
                    file = Files.get_file_by_id(file_id)
                    if file and file.path:
                        file_path = Storage.get_file(file.path)
                        content = await self.read_file_content(file_path)
                        await sandbox.upload_file(file.filename, content)
                        uploaded_filenames.append(file.filename)

            # 5. Execute Code
            files_before = set(await sandbox.list_files("."))

            # Run the code
            # For E2B, we need to handle the specific run_code that returns execution object
            # But our adapter returns ExecutionResult
            result = await sandbox.run_code(code, timeout=self.valves.TIMEOUT)

            # 6. Handle Generated Files
            files_after = set(await sandbox.list_files("."))
            new_files = files_after - files_before

            output_files = []
            image_files = []

            # Process new files on disk
            for filename in new_files:
                content = await sandbox.download_file(filename)
                if content:
                    file_meta = await self.save_file_to_storage(
                        filename, content, None, __user__
                    )
                    output_files.append(file_meta)
                    if file_meta["type"].startswith("image/"):
                        image_files.append(
                            {"type": "image", "url": file_meta["download_url"]}
                        )

            # Process in-memory images (E2B specific mostly)
            # Note: E2B images are already handled in E2BSandbox.run_code by saving them to the sandbox filesystem,
            # so they will be picked up by the file detection logic above (new_files).
            # No additional processing is needed here.

            # 7. Construct Response
            response_data = {
                "code": code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "results": result.results,
                "error": result.error,
                "files": output_files,
                "sandbox_id": result.sandbox_id,
            }

            # Emit images
            if image_files:
                await emit_files(__event_emitter__, image_files)

            # Final Status
            done_msg = f"done: {action_label}"
            if output_files:
                count = len(output_files)
                done_msg = f"done — {count} file{'s' if count > 1 else ''} created"

            await emit_status(
                done_msg,
                status="complete",
                emitter=__event_emitter__,
                done=True,
            )

            # Serialize
            output_str = json.dumps(response_data, indent=2)

            # Truncate if too large
            if len(output_str) > 500_000:
                response_data["stdout"] = "(truncated)"
                response_data["stderr"] = "(truncated)"
                output_str = json.dumps(response_data, indent=2)

            # Add persistence tags
            if result.sandbox_id:
                output_str += f"\n\n<!-- sandbox_id: {result.sandbox_id} -->"
                output_str += f"\n<!-- engine: {engine} -->"

            return output_str

        except Exception as e:
            logger.exception("Error during execution")
            await emit_status(
                "error during analysis",
                status="error",
                emitter=__event_emitter__,
                done=True,
                error=True,
            )
            return json.dumps({"error": str(e)})

        finally:
            # Stop/Pause sandbox
            # For E2B, this pauses. For Jupyter, we might want to keep it running if we want persistence,
            # but currently JupyterSandbox doesn't support true persistence across requests easily without a daemon.
            # However, we are reusing the session ID.
            await sandbox.stop()
