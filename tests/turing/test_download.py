"""Exercise real HTTP range resume and checksum failures on a small local file."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("turing_download", ROOT / "tools/turing_download.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)
DATA = bytes(range(256)) * 100


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".turing/tmp")
        self.root = Path(self.temp.name)
        self.original_root, self.original_pause = download.ROOT, download.PAUSE
        download.ROOT, download.PAUSE = self.root, self.root / "PAUSE"
        self.ranges = []
        self.ignore_range = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                start = int(self.headers["Range"].split("=")[1].split("-")[0])
                owner.ranges.append(start)
                self.send_response(200 if owner.ignore_range else 206)
                if not owner.ignore_range:
                    self.send_header("Content-Range", f"bytes {start}-{len(DATA)-1}/{len(DATA)}")
                self.end_headers()
                self.wfile.write(DATA if owner.ignore_range else DATA[start:])

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.item = {"url": f"http://127.0.0.1:{self.server.server_port}/model",
                     "path": "model.gguf", "size": len(DATA), "sha256": hashlib.sha256(DATA).hexdigest()}
        self.target = self.root / "model.gguf"
        self.partial = self.root / "model.gguf.part"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        download.ROOT, download.PAUSE = self.original_root, self.original_pause
        self.temp.cleanup()

    def test_resume_and_verify_existing_without_download(self):
        self.partial.write_bytes(DATA[:1234])
        download.download(self.item, 1e9)
        self.assertEqual(self.ranges, [1234])
        self.assertEqual(self.target.read_bytes(), DATA)
        self.assertFalse(self.partial.exists())
        download.download(self.item, 1e9)
        self.assertEqual(self.ranges, [1234])

    def test_corrupt_partial_never_published(self):
        self.partial.write_bytes(b"incorrect prefix")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            download.download(self.item, 1e9)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.partial.exists())

    def test_ignored_range_preserves_partial(self):
        self.partial.write_bytes(DATA[:1234])
        self.ignore_range = True
        with self.assertRaisesRegex(ValueError, "byte range"):
            download.download(self.item, 1e9)
        self.assertEqual(self.partial.read_bytes(), DATA[:1234])
        self.assertFalse(self.target.exists())

    def test_reject_path_outside_checkout(self):
        self.item["path"] = "../outside.gguf"
        with self.assertRaisesRegex(ValueError, "inside the checkout"):
            download.download(self.item, 1e9)
        self.assertEqual(self.ranges, [])


if __name__ == "__main__":
    unittest.main()
