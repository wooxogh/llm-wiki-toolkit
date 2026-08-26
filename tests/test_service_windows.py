from __future__ import annotations

import ctypes
import unittest
from unittest.mock import MagicMock, patch

from llm_wiki_v3 import service


class WindowsProcessProbeTests(unittest.TestCase):
    def test_windows_probe_uses_open_process_without_os_kill(self):
        kernel32 = MagicMock()
        kernel32.OpenProcess.return_value = 123
        with patch.object(service.os, "name", "nt"), patch.object(
            ctypes, "windll", MagicMock(kernel32=kernel32), create=True
        ):
            self.assertTrue(service.process_running(42))
        kernel32.OpenProcess.assert_called_once_with(0x1000, False, 42)
        kernel32.CloseHandle.assert_called_once_with(123)


if __name__ == "__main__":
    unittest.main()
